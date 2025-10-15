import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import json
import math
from typing import Any, Dict, List, Sequence

import torch

import g_eval as base_eval


_BASE_MAKE_MODEL_CACHE_KEY = base_eval.make_model_cache_key
_BASE_INITIALIZE_MODEL = base_eval.initialize_model
_BASE_COMPUTE_PROMPT_SCORE = base_eval.compute_prompt_score

def make_model_cache_key(cfg: Dict[str, Any]) -> str:
    provider = (cfg.get("provider") or "unsloth").lower()
    if provider != "vllm":
        return _BASE_MAKE_MODEL_CACHE_KEY(cfg)

    model_name = cfg.get("name") or cfg.get("model_name")
    if not model_name:
        raise KeyError("Model config requires 'name' or 'model_name'.")

    normalized: Dict[str, Any] = {
        "provider": "vllm",
        "name": model_name,
    }

    max_model_len = cfg.get("max_model_len") or cfg.get("max_seq_length")
    if max_model_len is not None:
        normalized["max_model_len"] = int(max_model_len)

    for key in (
        "tensor_parallel_size",
        "dtype",
        "revision",
        "download_dir",
        "gpu_memory_utilization",
        "swap_space",
        "quantization",
        "trust_remote_code",
        "enforce_eager",
    ):
        if key in cfg:
            normalized[key] = cfg[key]

    llm_kwargs = dict(cfg.get("llm_kwargs", {}))
    if llm_kwargs:
        normalized["llm_kwargs"] = llm_kwargs

    tokenizer_cfg = cfg.get("tokenizer") or {}
    if tokenizer_cfg:
        normalized["tokenizer"] = dict(tokenizer_cfg)

    return json.dumps(normalized, sort_keys=True, default=str)


def initialize_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    provider = (cfg.get("provider") or "unsloth").lower()
    if provider != "vllm":
        return _BASE_INITIALIZE_MODEL(cfg)

    try:
        from vllm import LLM  # type: ignore
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError(
            "vllm is not installed. Install it with `pip install vllm` to use provider='vllm'."
        ) from err

    model_name = cfg.get("name") or cfg.get("model_name")
    if not model_name:
        raise KeyError("Model config requires 'name' or 'model_name'.")

    constructor_kwargs: Dict[str, Any] = dict(cfg.get("llm_kwargs", {}))
    constructor_kwargs.setdefault("model", model_name)

    tokenizer_cfg = cfg.get("tokenizer") or {}
    tokenizer_name = tokenizer_cfg.get("name") or tokenizer_cfg.get("path")
    if tokenizer_name and "tokenizer" not in constructor_kwargs:
        constructor_kwargs["tokenizer"] = tokenizer_name
    if "mode" in tokenizer_cfg and "tokenizer_mode" not in constructor_kwargs:
        constructor_kwargs["tokenizer_mode"] = tokenizer_cfg["mode"]
    if "trust_remote_code" in tokenizer_cfg and "trust_remote_code" not in constructor_kwargs:
        constructor_kwargs["trust_remote_code"] = tokenizer_cfg["trust_remote_code"]

    for key in (
        "tensor_parallel_size",
        "dtype",
        "revision",
        "download_dir",
        "gpu_memory_utilization",
        "swap_space",
        "quantization",
        "trust_remote_code",
        "enforce_eager",
    ):
        if key in cfg and key not in constructor_kwargs:
            constructor_kwargs[key] = cfg[key]

    max_model_len = cfg.get("max_model_len") or cfg.get("max_seq_length")
    if max_model_len is not None and "max_model_len" not in constructor_kwargs:
        constructor_kwargs["max_model_len"] = int(max_model_len)

    llm = LLM(**constructor_kwargs)
    tokenizer = llm.get_tokenizer()

    setattr(llm, "_geval_provider", "vllm")
    setattr(tokenizer, "_geval_provider", "vllm")
    if not hasattr(llm, "eval"):
        setattr(llm, "eval", lambda *_, **__: None)

    return {
        "model": llm,
        "tokenizer": tokenizer,
        "provider": "vllm",
    }


def _build_variant_records(tokenizer: Any, choices: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    variant_records: List[Dict[str, Any]] = []
    for choice_index, choice in enumerate(choices):
        value = float(choice["value"])
        for variant_text in choice["variants"]:
            variant_ids = base_eval.encode_variant(tokenizer, variant_text)
            variant_records.append(
                {
                    "choice_index": choice_index,
                    "value": value,
                    "variant_text": variant_text,
                    "token_ids": variant_ids,
                }
            )

    if not variant_records:
        raise ValueError("No scoring variants were provided.")

    return variant_records


def _extract_vllm_logprob(logprob_entry: Any, token_id: int) -> float:
    if logprob_entry is None:
        raise ValueError("vLLM did not return prompt logprobs for the provided token.")

    if isinstance(logprob_entry, dict):
        candidate = logprob_entry.get(token_id)
        if candidate is None:
            for item in logprob_entry.values():
                candidate_id = getattr(item, "token_id", None)
                if candidate_id == token_id:
                    candidate = item
                    break
        logprob_entry = candidate

    if isinstance(logprob_entry, list):
        for candidate in logprob_entry:
            candidate_id = getattr(candidate, "token_id", None)
            if candidate_id is None and isinstance(candidate, (tuple, list)) and candidate:
                candidate_id = candidate[0]
            if candidate_id == token_id:
                if hasattr(candidate, "logprob"):
                    return float(candidate.logprob)
                if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
                    return float(candidate[1])
        raise ValueError(f"vLLM logprob data missing token {token_id}.")

    if hasattr(logprob_entry, "logprob"):
        return float(logprob_entry.logprob)

    if isinstance(logprob_entry, (tuple, list)):
        if len(logprob_entry) >= 2 and logprob_entry[0] == token_id:
            return float(logprob_entry[1])
        if len(logprob_entry) == 1:
            return float(logprob_entry[0])

    if isinstance(logprob_entry, (int, float)):
        return float(logprob_entry)

    raise ValueError(f"Unsupported vLLM logprob entry type for token {token_id!r}: {type(logprob_entry)!r}")


def _finalize_choice_scores(
    choices: Sequence[Dict[str, Any]],
    choice_variant_log_probs: Sequence[Sequence[float]],
    *,
    store_variant_details: bool,
    variant_details_bucket: Sequence[Sequence[Dict[str, Any]]] | None,
    store_value_log_probs: bool,
) -> Dict[str, Any]:
    value_log_probs: List[float] = []
    variant_details: List[Dict[str, Any]] = []

    for idx, (choice, per_variant_logs) in enumerate(zip(choices, choice_variant_log_probs)):
        if not per_variant_logs:
            raise ValueError(f"No valid tokenization variants found for value {choice['value']}.")
        log_prob_tensor = torch.tensor(per_variant_logs, dtype=torch.float32)
        combined_log_prob = float(torch.logsumexp(log_prob_tensor, dim=0).item())
        value_log_probs.append(combined_log_prob)

        if store_variant_details and variant_details_bucket is not None:
            variant_details.append(
                {
                    "value": float(choice["value"]),
                    "variants": list(variant_details_bucket[idx]),
                }
            )

    log_prob_tensor = torch.tensor(value_log_probs, dtype=torch.float32)
    log_total = float(torch.logsumexp(log_prob_tensor, dim=0))

    probabilities: Dict[str, float] = {}
    weighted_sum = 0.0
    for choice, value_log_prob in zip(choices, value_log_probs):
        prob = math.exp(value_log_prob - log_total)
        value = float(choice["value"])
        key = str(int(value) if value.is_integer() else value)
        probabilities[key] = prob
        weighted_sum += value * prob

    result: Dict[str, Any] = {
        "score": float(weighted_sum),
        "probabilities": probabilities,
    }

    if store_value_log_probs:
        value_log_prob_map = {}
        for choice, value_log_prob in zip(choices, value_log_probs):
            value = float(choice["value"])
            key = str(int(value) if value.is_integer() else value)
            value_log_prob_map[key] = float(value_log_prob)
        result["value_log_probs"] = value_log_prob_map

    if store_variant_details and variant_details:
        result["variant_details"] = variant_details

    return result


def _compute_prompt_score_vllm(
    llm: Any,
    tokenizer: Any,
    prompt_text: str,
    scoring_settings: Dict[str, Any],
) -> Dict[str, Any]:
    base_prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if isinstance(base_prompt_ids, torch.Tensor):
        base_prompt_ids = base_prompt_ids.tolist()
    base_prompt_ids = [int(token_id) for token_id in base_prompt_ids]

    choices = scoring_settings["choices"]
    store_variant_details = scoring_settings["store_variant_details"]
    store_value_log_probs = scoring_settings["store_value_log_probs"]

    variant_records = _build_variant_records(tokenizer, choices)

    combined_token_ids: List[List[int]] = []
    for record in variant_records:
        variant_ids = list(record["token_ids"])
        record["length"] = len(variant_ids)
        combined_ids = base_prompt_ids + variant_ids
        combined_token_ids.append(combined_ids)

    if not combined_token_ids:
        raise ValueError("No scoring variants were provided.")

    try:
        from vllm import SamplingParams  # type: ignore
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError(
            "vllm SamplingParams is unavailable; install vllm to use provider='vllm'."
        ) from err

    try:
        from vllm import PromptTokenIds  # type: ignore
        prompts = [PromptTokenIds(token_ids) for token_ids in combined_token_ids]
    except ImportError:
        prompts = [{"prompt_token_ids": token_ids} for token_ids in combined_token_ids]

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=1,
        prompt_logprobs=True,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    if len(outputs) != len(variant_records):
        raise RuntimeError("Mismatch between vLLM outputs and scoring variants.")

    choice_variant_log_probs: List[List[float]] = [[] for _ in choices]
    variant_details_bucket: List[List[Dict[str, Any]]] | None = None
    if store_variant_details:
        variant_details_bucket = [[] for _ in choices]

    for record, output in zip(variant_records, outputs):
        length = int(record.get("length", 0))
        if length <= 0:
            continue

        prompt_token_ids = list(getattr(output, "prompt_token_ids", []) or [])
        prompt_logprobs = list(getattr(output, "prompt_logprobs", []) or [])
        if len(prompt_token_ids) < length or len(prompt_logprobs) < length:
            raise ValueError("vLLM prompt output shorter than requested variant tokens.")

        total_log_prob = 0.0
        for token_id, logprob_entry in zip(prompt_token_ids[-length:], prompt_logprobs[-length:]):
            total_log_prob += _extract_vllm_logprob(logprob_entry, token_id)

        choice_idx = record["choice_index"]
        choice_variant_log_probs[choice_idx].append(total_log_prob)

        if variant_details_bucket is not None:
            variant_details_bucket[choice_idx].append(
                {
                    "text": record["variant_text"],
                    "log_prob": total_log_prob,
                }
            )

    return _finalize_choice_scores(
        choices,
        choice_variant_log_probs,
        store_variant_details=store_variant_details,
        variant_details_bucket=variant_details_bucket,
        store_value_log_probs=store_value_log_probs,
    )


def compute_prompt_score(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    scoring_settings: Dict[str, Any],
) -> Dict[str, Any]:
    provider = getattr(model, "_geval_provider", None) or getattr(tokenizer, "_geval_provider", None)
    if provider == "vllm":
        return _compute_prompt_score_vllm(model, tokenizer, prompt_text, scoring_settings)
    return _BASE_COMPUTE_PROMPT_SCORE(model, tokenizer, prompt_text, scoring_settings)


base_eval.make_model_cache_key = make_model_cache_key
base_eval.initialize_model = initialize_model
base_eval.compute_prompt_score = compute_prompt_score


def main(
    config_path: str = "geval_vllm.yaml",
    limit: int | None = None,
    prompt_names: Sequence[str] | None = None,
    list_prompts: bool = False,
) -> None:
    base_eval.main(
        config_path=config_path,
        limit=limit,
        prompt_names=prompt_names,
        list_prompts=list_prompts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run G-Eval (vLLM provider) over augmented datasets.")
    parser.add_argument("--config", default="geval_vllm.yaml", help="Path to the G-Eval config file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples per dataset.")
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Specify prompt name(s) to run. Can be provided multiple times.",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available G-Eval prompt names and exit.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        limit=args.limit,
        prompt_names=args.prompts,
        list_prompts=args.list_prompts,
    )
