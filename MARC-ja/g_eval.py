import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# 合成データをG-Eval手法で多観点スコアリングするスクリプト

from augment import (
    get_model_device,
    initialize_model,
    load_config,
    load_subset_dataset,
    make_model_cache_key,
    render_chat_prompts,
    resolve_prompt_selection,
    set_seed,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdmは任意依存
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


# デフォルトのスコア選択肢（1〜5）を定義
DEFAULT_SCORE_CHOICES: List[Dict[str, Any]] = [
    {"value": 1, "variants": ["1", " 1"]},
    {"value": 2, "variants": ["2", " 2"]},
    {"value": 3, "variants": ["3", " 3"]},
    {"value": 4, "variants": ["4", " 4"]},
    {"value": 5, "variants": ["5", " 5"]},
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def dumps_json(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, default=_json_default)


def make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, tuple):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(make_hashable(v) for v in value))
    return value


def normalize_score_choices(scoring_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """YAMLの設定からスコア選択肢と詳細保存フラグを正規化する。"""
    choices_cfg = scoring_cfg.get("choices")
    if not choices_cfg:
        choices = list(DEFAULT_SCORE_CHOICES)
    else:
        choices = []
        for entry in choices_cfg:
            if "value" not in entry:
                raise ValueError("Each scoring.choices entry requires 'value'.")
            value = float(entry["value"])
            variants = entry.get("variants") or entry.get("tokens") or entry.get("texts")
            if not variants:
                variants = [str(int(value))]
            normalized_variants: List[str] = []
            for variant in variants:
                if not isinstance(variant, str):
                    raise ValueError("scoring.choices.variant entries must be strings.")
                if variant not in normalized_variants:
                    normalized_variants.append(variant)
            choices.append({"value": value, "variants": normalized_variants})

    return {
        "choices": choices,
        "store_variant_details": bool(scoring_cfg.get("store_variant_details", False)),
        "store_value_log_probs": bool(scoring_cfg.get("store_value_log_probs", False)),
    }


def initialize_wandb_run(eval_cfg: Dict[str, Any], config_path: Path) -> Any | None:
    """W&BのRunを初期化する。無効時はNoneを返す。"""
    wandb_cfg = eval_cfg.get("wandb") or {}
    if wandb_cfg.get("enabled", True) is False:
        return None

    try:
        import wandb  # type: ignore
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError(
            "wandb logging is enabled but the wandb package is not installed. "
            "Install it with `pip install wandb` or disable logging via geval.wandb.enabled=false."
        ) from err

    init_kwargs: Dict[str, Any] = {
        "project": wandb_cfg.get("project") or "geval",
        "config": eval_cfg,
    }

    entity = wandb_cfg.get("entity")
    if entity:
        init_kwargs["entity"] = entity

    run_name = wandb_cfg.get("run_name") or f"geval-{config_path.stem}"
    if run_name:
        init_kwargs["name"] = run_name

    tags = wandb_cfg.get("tags")
    if tags:
        init_kwargs["tags"] = list(tags)

    mode = wandb_cfg.get("mode")
    if mode:
        init_kwargs["mode"] = mode

    dir_path = wandb_cfg.get("dir")
    if dir_path:
        init_kwargs["dir"] = dir_path

    return wandb.init(**init_kwargs)


def encode_variant(tokenizer: Any, text: str) -> List[int]:
    """各スコア候補のトークン列をTokenizerで符号化する。"""
    attempts = [text]
    if not text.startswith(" "):
        attempts.append(" " + text)
    if not text.startswith("\n"):
        attempts.append("\n" + text)
    attempts.append(text.strip())

    for attempt in attempts:
        tokens = tokenizer.encode(attempt, add_special_tokens=False)
        if tokens:
            return tokens

    tokens_with_special = tokenizer.encode(text, add_special_tokens=True)
    if tokens_with_special:
        trimmed = [tid for tid in tokens_with_special if tid not in getattr(tokenizer, "all_special_ids", [])]
        if trimmed:
            return trimmed

    raise ValueError(f"Tokenizer could not encode variant '{text}'.")


def compute_prompt_score(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    scoring_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """単一プロンプトに対してスコアと各確率情報を計算する。"""
    model_device = get_model_device(model)
    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_inputs = {k: v.to(model_device) for k, v in prompt_inputs.items()}

    base_input_ids = prompt_inputs["input_ids"]
    base_attention_mask = prompt_inputs.get("attention_mask")

    choices = scoring_settings["choices"]
    store_variant_details = scoring_settings["store_variant_details"]
    store_value_log_probs = scoring_settings["store_value_log_probs"]

    if base_input_ids.shape[0] != 1:
        raise ValueError("compute_prompt_score expects a single prompt example per batch.")

    base_prompt_ids = base_input_ids[0]
    base_prompt_len = base_prompt_ids.shape[0]

    if base_attention_mask is not None:
        base_mask = base_attention_mask[0]
        if base_mask.dtype != torch.long:
            base_mask = base_mask.to(dtype=torch.long)
    else:
        base_mask = torch.ones(base_prompt_len, dtype=torch.long, device=model_device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer requires pad_token_id or eos_token_id to batch variants.")

    variant_records: List[Dict[str, Any]] = []
    for choice_index, choice in enumerate(choices):
        value = float(choice["value"])
        for variant_text in choice["variants"]:
            variant_ids = encode_variant(tokenizer, variant_text)
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

    full_input_sequences: List[torch.Tensor] = []
    full_attention_sequences: List[torch.Tensor] = []
    for record in variant_records:
        variant_tensor = torch.tensor(record["token_ids"], dtype=base_prompt_ids.dtype, device=model_device)
        record["tensor"] = variant_tensor
        record["length"] = variant_tensor.numel()

        full_input_sequences.append(torch.cat([base_prompt_ids, variant_tensor], dim=0))

        variant_mask = torch.ones(record["length"], dtype=base_mask.dtype, device=model_device)
        full_attention_sequences.append(torch.cat([base_mask, variant_mask], dim=0))

    batched_input_ids = pad_sequence(
        full_input_sequences,
        batch_first=True,
        padding_value=int(pad_token_id),
    )
    batched_attention_mask = pad_sequence(
        full_attention_sequences,
        batch_first=True,
        padding_value=0,
    )

    with torch.no_grad():
        outputs = model(input_ids=batched_input_ids, attention_mask=batched_attention_mask)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)

    choice_variant_log_probs: List[List[float]] = [[] for _ in choices]
    variant_details_bucket: List[List[Dict[str, Any]]] | None = None
    if store_variant_details:
        variant_details_bucket = [[] for _ in choices]

    for batch_index, record in enumerate(variant_records):
        length = record["length"]
        if length == 0:
            continue

        positions = torch.arange(length, device=model_device, dtype=torch.long) + (base_prompt_len - 1)
        token_ids_tensor = record["tensor"]
        token_log_probs = log_probs[batch_index, positions, token_ids_tensor]
        total_log_prob = float(token_log_probs.sum().item())

        choice_idx = record["choice_index"]
        choice_variant_log_probs[choice_idx].append(total_log_prob)

        if variant_details_bucket is not None:
            variant_details_bucket[choice_idx].append(
                {
                    "text": record["variant_text"],
                    "log_prob": total_log_prob,
                }
            )

    value_log_probs: List[float] = []
    variant_details: List[Dict[str, Any]] = []

    for choice_index, (choice, per_variant_logs) in enumerate(zip(choices, choice_variant_log_probs)):
        if not per_variant_logs:
            raise ValueError(f"No valid tokenization variants found for value {choice['value']}.")

        log_prob_tensor = torch.tensor(per_variant_logs, dtype=torch.float32, device=model_device)
        combined_log_prob = float(torch.logsumexp(log_prob_tensor, dim=0).item())
        value_log_probs.append(combined_log_prob)

        if variant_details_bucket is not None:
            variant_details.append(
                {
                    "value": float(choice["value"]),
                    "variants": variant_details_bucket[choice_index],
                }
            )

    log_prob_tensor = torch.tensor(value_log_probs, dtype=torch.float32)
    log_total = float(torch.logsumexp(log_prob_tensor, dim=0))

    probabilities: Dict[str, float] = {}
    weighted_sum = 0.0
    for choice, value_log_prob in zip(choices, value_log_probs):
        prob = math.exp(value_log_prob - log_total)
        value = float(choice["value"])
        probabilities[str(int(value) if value.is_integer() else value)] = prob
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

    if store_variant_details:
        result["variant_details"] = variant_details

    return result


def build_prompt_messages(
    prompt_name: str,
    prompt_cfg: Dict[str, Any],
    example: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """データレコードと設定から評価用のメッセージを組み立てる。"""
    user_template = prompt_cfg.get("user_prompt_template")
    if not user_template:
        raise ValueError(f"Prompt '{prompt_name}' requires user_prompt_template.")

    text_column = dataset_cfg.get("text_column") or "sentence"
    if text_column not in example:
        raise KeyError(f"Prompt '{prompt_name}' requires column '{text_column}' in the dataset.")
    text_value = example[text_column]

    label_column = dataset_cfg.get("label_column")
    label_names = list(dataset_cfg.get("label_names") or [])

    format_values: Dict[str, Any] = {
        "input_text": text_value,
        "generated_text": text_value,
    }

    if label_column and label_column in example:
        label_value = example[label_column]
        format_values["label_id"] = label_value
        label_idx: int | None = None
        try:
            label_idx = int(label_value)
        except Exception:
            label_idx = None
        if label_idx is not None and 0 <= label_idx < len(label_names):
            format_values["label_name"] = label_names[label_idx]
        else:
            format_values["label_name"] = label_value

    template_variables = prompt_cfg.get("template_variables") or {}
    for placeholder, column_name in template_variables.items():
        if isinstance(column_name, str) and column_name.startswith("$"):
            special = column_name[1:]
            if special == "label_name" and "label_name" in format_values:
                format_values[placeholder] = format_values["label_name"]
            elif special == "label_id" and "label_id" in format_values:
                format_values[placeholder] = format_values["label_id"]
            else:
                raise KeyError(
                    f"Prompt '{prompt_name}' template_variables placeholder '{placeholder}' "
                    f"refers to unknown special value '{column_name}'."
                )
        else:
            if column_name not in example:
                raise KeyError(
                    f"Prompt '{prompt_name}' template_variables placeholder '{placeholder}' "
                    f"requires column '{column_name}' in the dataset."
                )
            format_values[placeholder] = example[column_name]

    template_defaults = prompt_cfg.get("template_defaults") or {}
    for key, value in template_defaults.items():
        format_values.setdefault(key, value)

    try:
        rendered_user = user_template.format(**format_values)
    except KeyError as err:
        raise KeyError(
            f"Prompt '{prompt_name}' user_prompt_template is missing key '{err.args[0]}'."
        ) from err

    system_prompt = prompt_cfg.get("system_prompt") or None

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": rendered_user}],
        }
    )
    return messages


def resolve_dataset_entries(
    base_dataset_cfg: Dict[str, Any],
    prompt_name: str,
    prompt_cfg: Dict[str, Any],
    dataset_cache: Dict[Tuple[Path, int | None], Any],
    limit: int | None,
) -> List[Dict[str, Any]]:
    """プロンプトごとのデータセット入力設定を解決し、キャッシュを利用して読み込む。"""
    merged_dataset_cfg = dict(base_dataset_cfg)
    prompt_dataset_cfg = prompt_cfg.get("dataset") or {}
    merged_dataset_cfg.update(prompt_dataset_cfg)

    paths_value = merged_dataset_cfg.get("input_paths")
    raw_paths: List[Any]
    if paths_value:
        if isinstance(paths_value, (str, Path)):
            raw_paths = [paths_value]
        else:
            raw_paths = list(paths_value)
    else:
        input_path = merged_dataset_cfg.get("input_path")
        if not input_path:
            raise ValueError(f"Prompt '{prompt_name}' requires dataset.input_path when input_paths is not provided.")
        raw_paths = [input_path]

    dataset_entries: List[Dict[str, Any]] = []
    for order, raw_path in enumerate(raw_paths):
        dataset_path = Path(raw_path)
        cache_key = (dataset_path.resolve(), limit)
        if cache_key not in dataset_cache:
            dataset_obj = load_subset_dataset(dataset_path)
            if limit is not None:
                dataset_obj = dataset_obj.select(range(min(limit, len(dataset_obj))))
            dataset_cache[cache_key] = dataset_obj
        entry_cfg = dict(merged_dataset_cfg)
        entry_cfg["input_path"] = str(dataset_path)
        entry_cfg.pop("input_paths", None)
        dataset_entries.append(
            {
                "dataset": dataset_cache[cache_key],
                "dataset_cfg": entry_cfg,
                "dataset_path": dataset_path,
                "dataset_name": dataset_path.stem,
                "dataset_index": order + 1,
            }
        )

    return dataset_entries


def format_output_path(
    template: str | None,
    dataset_name: str,
    dataset_index: int,
    *,
    prompt_name: str | None = None,
) -> Path | None:
    """出力テンプレートから実際のファイルパスを生成、必要なら親ディレクトリ作成。"""
    if not template:
        return None
    format_values = {
        "dataset_name": dataset_name,
        "dataset_index": dataset_index,
    }
    if prompt_name:
        format_values["prompt_name"] = prompt_name
    try:
        formatted = template.format(**format_values)
    except KeyError as err:
        raise KeyError(
            f"Output path template is missing key '{err.args[0]}' in format values."
        ) from err
    path = Path(formatted)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_dataset_group(
    group_key: Tuple[Any, ...],
    group_payload: Dict[str, Any],
    *,
    model_cfg: Dict[str, Any],
    scoring_settings: Dict[str, Any],
    output_cfg: Dict[str, Any],
    model_cache: Dict[str, Dict[str, Any]],
    wandb_run: Any | None = None,
    wandb_state: Dict[str, Any] | None = None,
) -> None:
    dataset = group_payload["dataset"]
    dataset_cfg = group_payload["dataset_cfg"]
    prompts = group_payload["prompts"]
    dataset_name = group_payload["dataset_name"]
    dataset_index = group_payload["dataset_index"]

    scored_records: List[Dict[str, Any]] = []
    per_prompt_records: Dict[str, List[Dict[str, Any]]] = {}
    prompt_score_sums: Dict[str, float] = {}
    prompt_score_counts: Dict[str, int] = {}

    progress_desc = f"G-Eval ({dataset_name})"
    iterable: Iterable[Dict[str, Any]] = dataset
    dataset_size = len(dataset)
    wandb_enabled = wandb_run is not None
    if wandb_enabled and wandb_state is None:
        wandb_state = {"step": 0}
    if wandb_state is not None:
        wandb_state["step"] = int(wandb_state.get("step", 0))

    for example in tqdm(iterable, desc=progress_desc, total=len(dataset)):
        example_dict = {key: example[key] for key in example.keys()}

        per_prompt_results: Dict[str, Dict[str, Any]] = {}
        prompt_scores: List[float] = []

        for prompt_entry in prompts:
            prompt_name = prompt_entry["name"]
            prompt_cfg = prompt_entry["cfg"]
            prompt_dataset_cfg = prompt_entry["dataset_cfg"]

            effective_model_cfg = dict(model_cfg)
            prompt_model_cfg = prompt_cfg.get("model") or {}

            from_pretrained_base = dict(model_cfg.get("from_pretrained", {}))
            from_pretrained_override = dict(prompt_model_cfg.get("from_pretrained", {}))

            for key, value in prompt_model_cfg.items():
                if key == "from_pretrained":
                    continue
                effective_model_cfg[key] = value

            if from_pretrained_base or from_pretrained_override:
                merged_from_pretrained = dict(from_pretrained_base)
                merged_from_pretrained.update(from_pretrained_override)
                effective_model_cfg["from_pretrained"] = merged_from_pretrained

            model_cache_key = make_model_cache_key(effective_model_cfg)
            if model_cache_key not in model_cache:
                model_cache[model_cache_key] = initialize_model(effective_model_cfg)
            model_bundle = model_cache[model_cache_key]
            model = model_bundle["model"]
            tokenizer = model_bundle["tokenizer"]
            model.eval()

            messages = build_prompt_messages(prompt_name, prompt_cfg, example_dict, prompt_dataset_cfg)
            prompt_text = render_chat_prompts(tokenizer, [messages])[0]

            prompt_result = compute_prompt_score(model, tokenizer, prompt_text, scoring_settings)

            per_prompt_results[prompt_name] = prompt_result
            prompt_scores.append(prompt_result["score"])
            prompt_score_sums[prompt_name] = prompt_score_sums.get(prompt_name, 0.0) + float(prompt_result["score"])
            prompt_score_counts[prompt_name] = prompt_score_counts.get(prompt_name, 0) + 1

            prompt_specific_record = dict(example_dict)
            prompt_specific_record["geval"] = {
                "prompt": prompt_name,
                "score": prompt_result["score"],
                "probabilities": prompt_result["probabilities"],
            }
            if "value_log_probs" in prompt_result:
                prompt_specific_record["geval"]["value_log_probs"] = prompt_result["value_log_probs"]
            if "variant_details" in prompt_result:
                prompt_specific_record["geval"]["variant_details"] = prompt_result["variant_details"]

            per_prompt_records.setdefault(prompt_name, []).append(prompt_specific_record)

        if prompt_scores:
            aggregate_score = float(sum(prompt_scores) / len(prompt_scores))
        else:
            aggregate_score = None

        aggregate_info = {
            "score": aggregate_score,
        }

        if wandb_enabled and wandb_state is not None:
            example_index = len(scored_records) + 1
            progress_ratio = float(example_index / dataset_size) if dataset_size else 0.0
            step_value = int(wandb_state["step"])
            log_payload: Dict[str, Any] = {
                "dataset/name": dataset_name,
                "dataset/index": dataset_index,
                "dataset/example_index": example_index,
                "dataset/progress": progress_ratio,
            }
            if aggregate_score is not None:
                log_payload["geval/aggregate_score"] = float(aggregate_score)
            for prompt_name, prompt_result in per_prompt_results.items():
                log_payload[f"geval/prompts/{prompt_name}/score"] = float(prompt_result["score"])
            wandb_run.log(log_payload, step=step_value)
            wandb_state["step"] = step_value + 1

        scored_record = dict(example_dict)
        scored_record["geval"] = {
            "per_prompt": per_prompt_results,
            "aggregate": aggregate_info,
        }
        scored_records.append(scored_record)

    dataset_scores_path = format_output_path(
        output_cfg.get("dataset_scores_path_template") or output_cfg.get("scored_path_template"),
        dataset_name,
        dataset_index,
    )
    if dataset_scores_path:
        with dataset_scores_path.open("w", encoding="utf-8") as f:
            for record in scored_records:
                f.write(dumps_json(record) + "\n")

    prompt_template = output_cfg.get("prompt_scores_path_template")
    if prompt_template:
        for prompt_name, prompt_records in per_prompt_records.items():
            prompt_path = format_output_path(
                prompt_template,
                dataset_name,
                dataset_index,
                prompt_name=prompt_name,
            )
            if prompt_path:
                with prompt_path.open("w", encoding="utf-8") as f:
                    for record in prompt_records:
                        f.write(dumps_json(record) + "\n")

    if wandb_enabled and wandb_state is not None:
        summary_payload: Dict[str, Any] = {
            "dataset/name": dataset_name,
            "dataset/index": dataset_index,
            "dataset/examples": len(scored_records),
        }
        aggregate_scores = [
            record["geval"]["aggregate"]["score"]
            for record in scored_records
            if record["geval"]["aggregate"]["score"] is not None
        ]
        if aggregate_scores:
            summary_payload["geval/dataset_mean_score"] = float(np.mean(aggregate_scores))
            summary_payload["geval/dataset_std_score"] = float(np.std(aggregate_scores))
            summary_payload["geval/dataset_min_score"] = float(min(aggregate_scores))
            summary_payload["geval/dataset_max_score"] = float(max(aggregate_scores))
        for prompt_name, total in prompt_score_sums.items():
            count = prompt_score_counts.get(prompt_name, 0)
            if count:
                summary_payload[f"geval/prompts/{prompt_name}/mean_score"] = float(total / count)
        step_value = int(wandb_state["step"])
        wandb_run.log(summary_payload, step=step_value)
        wandb_state["step"] = step_value + 1


def main(
    config_path: str = "geval.yaml",
    limit: int | None = None,
    prompt_names: Sequence[str] | None = None,
    list_prompts: bool = False,
) -> None:
    """G-Eval全体のエントリポイント。設定読込・データグループ作成・評価実行を担う。"""
    config_path_obj = Path(config_path)
    config = load_config(config_path_obj)
    eval_cfg = config.get("geval") or {}

    dataset_cfg = eval_cfg.get("dataset") or {}
    model_cfg = eval_cfg.get("model") or {}
    scoring_settings = normalize_score_choices(eval_cfg.get("scoring") or {})
    output_cfg = eval_cfg.get("output") or {}

    seed = eval_cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    prompts_cfg = eval_cfg.get("prompts") or {}

    if list_prompts:
        if not prompts_cfg:
            print("[Info] No prompts configured under geval.prompts.")
            return
        print("[Info] Registered G-Eval prompts:")
        for name, prompt in prompts_cfg.items():
            description = prompt.get("description")
            if description:
                print(f"  - {name}: {description}")
            else:
                print(f"  - {name}")
        return

    selected_prompts = resolve_prompt_selection(prompts_cfg, prompt_names)
    if not selected_prompts:
        print("[Warning] No prompts selected for evaluation.")
        return

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive if provided.")

    dataset_cache: Dict[Tuple[Path, int | None], Any] = {}

    # 同一データセット＋設定組み合わせでプロンプト群を束ね、重複処理を防ぐ
    dataset_groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for prompt_name, prompt_cfg in selected_prompts:
        dataset_entries = resolve_dataset_entries(
            dataset_cfg,
            prompt_name,
            prompt_cfg,
            dataset_cache,
            limit,
        )
        for entry in dataset_entries:
            dataset_path = entry["dataset_path"].resolve()
            dataset_specific_cfg = entry["dataset_cfg"]
            key = (
                dataset_path,
                make_hashable(dataset_specific_cfg),
            )
            if key not in dataset_groups:
                dataset_groups[key] = {
                    "dataset": entry["dataset"],
                    "dataset_cfg": dataset_specific_cfg,
                    "dataset_path": dataset_path,
                    "dataset_name": entry["dataset_name"],
                    "dataset_index": entry["dataset_index"],
                    "prompts": [],
                }
            dataset_groups[key]["prompts"].append(
                {
                    "name": prompt_name,
                    "cfg": prompt_cfg,
                    "dataset_cfg": dataset_specific_cfg,
                }
            )

    if not dataset_groups:
        print("[Warning] No datasets matched the selected prompts.")
        return

    wandb_run = initialize_wandb_run(eval_cfg, config_path_obj)
    wandb_state: Dict[str, Any] | None = {"step": 0} if wandb_run is not None else None

    model_cache: Dict[str, Dict[str, Any]] = {}

    try:
        for key, payload in dataset_groups.items():
            evaluate_dataset_group(
                key,
                payload,
                model_cfg=model_cfg,
                scoring_settings=scoring_settings,
                output_cfg=output_cfg,
                model_cache=model_cache,
                wandb_run=wandb_run,
                wandb_state=wandb_state,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run G-Eval filtering over augmented datasets.")
    parser.add_argument("--config", default="geval.yaml", help="Path to the G-Eval config file.")
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
