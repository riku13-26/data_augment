import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from datasets import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdmは任意依存
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


GENERATION_PARAM_KEYS = {
    "max_new_tokens",
    "temperature",
    "top_p",
    "repetition_penalty",
    "do_sample",
    "seed",
    "num_generations_per_sample",
    "top_k",
    "batch_size",
}


def infer_chat_template_name(model_name: str) -> str | None:
    lowered = model_name.lower()
    if "gemma-3" in lowered or "gemma3" in lowered:
        return "gemma-3"
    if "gemma-2" in lowered or "gemma2" in lowered:
        return "gemma2"
    if "gemma" in lowered:
        return "gemma"
    if "llama-3" in lowered or "llama3" in lowered:
        return "llama-3"
    if "llama-2" in lowered or "llama2" in lowered:
        return "llama-2"
    if "llama" in lowered:
        return "llama-2"
    if "qwen3" in lowered or "qwen-3" in lowered:
        return "qwen3"
    if "qwen2" in lowered or "qwen-2" in lowered:
        return "qwen2"
    if "mistral" in lowered:
        return "mistral"
    if "phi-4" in lowered or "phi4" in lowered:
        return "phi-4"
    if "phi-3" in lowered or "phi3" in lowered:
        return "phi-3"
    if "gpt-oss" in lowered or "gpt_oss" in lowered:
        return "gpt-oss"
    return None


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subset_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {path}. Ensure the path is correct or generate the dataset first."
        )
    return Dataset.from_json(str(path))


def make_model_cache_key(cfg: Dict[str, Any]) -> str:
    model_name = cfg.get("name") or cfg.get("model_name")
    if not model_name:
        raise KeyError("Model config requires 'name' or 'model_name'.")

    provider = (cfg.get("provider") or "unsloth").lower()

    normalized: Dict[str, Any] = {
        "provider": provider,
        "name": model_name,
        "max_seq_length": int(cfg.get("max_seq_length", 4096)),
        "device_map": cfg.get("device_map", "auto"),
        "from_pretrained": dict(cfg.get("from_pretrained", {})),
    }

    for key in ("load_in_4bit", "load_in_8bit", "torch_dtype"):
        if key in cfg and key not in normalized["from_pretrained"]:
            normalized["from_pretrained"][key] = cfg[key]

    for key in ("quantization", "attn_implementation", "flash_attention_2", "chat_template", "use_fast_tokenizer"):
        if key in cfg:
            normalized[key] = cfg[key]

    tokenizer_cfg = cfg.get("tokenizer") or {}
    if tokenizer_cfg:
        normalized["tokenizer"] = dict(tokenizer_cfg)

    return json.dumps(normalized, sort_keys=True, default=str)



def resolve_torch_dtype(dtype_value: Any) -> torch.dtype | None:
    if dtype_value is None:
        return None
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if isinstance(dtype_value, str):
        name = dtype_value.strip().lower()
        alias = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        if name in alias:
            return alias[name]
        if hasattr(torch, name):
            candidate = getattr(torch, name)
            if isinstance(candidate, torch.dtype):
                return candidate
    raise ValueError(f"Unsupported torch_dtype value: {dtype_value!r}")



def get_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    try:
        parameter = next(model.parameters())
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")
    return parameter.device



def render_chat_prompts(tokenizer: Any, prompts: Sequence[List[Dict[str, Any]]]) -> List[str]:
    rendered: List[str] = []
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)

    for messages in prompts:
        rendered_text: str | None = None
        if callable(apply_chat_template):
            try:
                rendered_text = apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered_text = None

        if rendered_text is None:
            parts: List[str] = []
            for message in messages:
                role = message.get("role") or ""
                content_items = message.get("content") or []
                text_fragments: List[str] = []
                for item in content_items:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_fragments.append(item.get("text", ""))
                        else:
                            text_fragments.append(str(item))
                    else:
                        text_fragments.append(str(item))
                text = "\n\n".join(fragment for fragment in text_fragments if fragment).strip()
                if not text:
                    continue
                if role == "system":
                    parts.append(text)
                elif role == "assistant":
                    parts.append(f"Assistant:\n{text}")
                elif role == "user":
                    parts.append(text)
                else:
                    parts.append(f"{role.capitalize()}:\n{text}")
            rendered_text = "\n\n".join(parts).strip()
        rendered.append((rendered_text or "").rstrip() + "\n\n")

    return rendered

def initialize_unsloth_model(cfg: Dict[str, Any]):
    try:
        from unsloth import FastLanguageModel
        try:
            from unsloth.chat_templates import get_chat_template as _get_chat_template
        except Exception:
            try:
                from unsloth import get_chat_template as _get_chat_template  # type: ignore
            except Exception:
                _get_chat_template = None
    except ImportError as err:
        raise ImportError(
            "unsloth がインストールされていません。pip install unsloth で追加してください。"
        ) from err

    model_name = cfg.get("name") or cfg.get("model_name")
    if not model_name:
        raise KeyError("Model config requires 'name' or 'model_name'.")

    max_seq_length = int(cfg.get("max_seq_length", 4096))
    device_map = cfg.get("device_map", "auto")

    from_pretrained_kwargs = dict(cfg.get("from_pretrained", {}))
    for key in ("load_in_4bit", "load_in_8bit", "torch_dtype"):
        if key in cfg and key not in from_pretrained_kwargs:
            from_pretrained_kwargs[key] = cfg[key]

    load_in_4bit = bool(from_pretrained_kwargs.get("load_in_4bit", False))
    if load_in_4bit and str(device_map) in {"auto", "balanced", "balanced_low_0"}:
        device_map = 0

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        device_map=device_map,
        **from_pretrained_kwargs,
    )

    FastLanguageModel.for_inference(model)
    model.eval()

    chat_template_name = cfg.get("chat_template") or infer_chat_template_name(model_name)
    if chat_template_name and _get_chat_template is not None:
        try:
            tokenizer = _get_chat_template(tokenizer, chat_template=chat_template_name)
        except Exception as err:
            print(f"[Warning] Failed to apply chat template '{chat_template_name}': {err}")
    elif chat_template_name and _get_chat_template is None:
        print(f"[Warning] unsloth.get_chat_template が見つからないため、テンプレート '{chat_template_name}' は未適用です。")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        if getattr(model, "generation_config", None) is not None:
            gc = model.generation_config
            if gc.pad_token_id is None:
                gc.pad_token_id = tokenizer.pad_token_id
            if gc.eos_token_id is None:
                gc.eos_token_id = tokenizer.eos_token_id
        if getattr(model, "config", None) is not None:
            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    return model, tokenizer



def initialize_hf_model(cfg: Dict[str, Any]):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as err:
        raise ImportError(
            "transformers がインストールされていません。pip install transformers で追加してください。"
        ) from err

    model_name = cfg.get("name") or cfg.get("model_name")
    if not model_name:
        raise KeyError("Model config requires 'name' or 'model_name'.")

    device_map = cfg.get("device_map", "auto")
    max_seq_length = cfg.get("max_seq_length")

    from_pretrained_kwargs = dict(cfg.get("from_pretrained", {}))

    legacy_load_in_4bit = bool(from_pretrained_kwargs.pop("load_in_4bit", False) or cfg.get("load_in_4bit"))
    legacy_load_in_8bit = bool(from_pretrained_kwargs.pop("load_in_8bit", False) or cfg.get("load_in_8bit"))
    if legacy_load_in_4bit and legacy_load_in_8bit:
        raise ValueError(
            "Hugging Face provider cannot enable both load_in_4bit and load_in_8bit. Please choose one."
        )

    if "torch_dtype" in from_pretrained_kwargs:
        from_pretrained_kwargs["torch_dtype"] = resolve_torch_dtype(from_pretrained_kwargs["torch_dtype"])
    elif "torch_dtype" in cfg:
        from_pretrained_kwargs["torch_dtype"] = resolve_torch_dtype(cfg["torch_dtype"])

    quantization_config_present = "quantization_config" in from_pretrained_kwargs
    quantization_value = None
    quantization_source = None
    quantization_setting = cfg.get("quantization")
    if quantization_setting:
        quantization_value = str(quantization_setting).lower()
        quantization_source = "explicit"
    elif not quantization_config_present:
        if legacy_load_in_4bit:
            quantization_value = "4bit"
            quantization_source = "legacy"
        elif legacy_load_in_8bit:
            quantization_value = "8bit"
            quantization_source = "legacy"

    if quantization_value and not quantization_config_present:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as err:
            raise ImportError(
                "quantization に bitsandbytes を利用するには pip install bitsandbytes が必要です。"
            ) from err

        if quantization_value in {"4bit", "bnb-4bit", "bitsandbytes_4bit"}:
            compute_dtype = resolve_torch_dtype(
                cfg.get("quantization_compute_dtype")
                or cfg.get("bnb_4bit_compute_dtype")
                or torch.float16
            )
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            )
        elif quantization_value in {"8bit", "bnb-8bit", "bitsandbytes_8bit"}:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"Unsupported quantization mode: {quantization_setting!r}")
        from_pretrained_kwargs["quantization_config"] = quant_config
        if quantization_source == "legacy":
            print("[Info] Converted legacy load_in_4bit/load_in_8bit flags to quantization_config for Hugging Face provider.")

    from_pretrained_kwargs.setdefault("device_map", device_map)

    attn_impl = cfg.get("attn_implementation")
    if cfg.get("flash_attention_2"):
        attn_impl = "flash_attention_2"
    if attn_impl:
        from_pretrained_kwargs.setdefault("attn_implementation", attn_impl)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **from_pretrained_kwargs,
    )
    model.eval()

    if attn_impl:
        try:
            model.config.attn_implementation = attn_impl
        except Exception:
            pass

    tokenizer_kwargs = dict(cfg.get("tokenizer", {}))
    if "use_fast" not in tokenizer_kwargs and "use_fast_tokenizer" in cfg:
        tokenizer_kwargs["use_fast"] = bool(cfg.get("use_fast_tokenizer", True))

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    if max_seq_length:
        import contextlib as _contextlib

        with _contextlib.suppress(Exception):
            tokenizer.model_max_length = int(max_seq_length)
            if getattr(model, "config", None) is not None:
                if hasattr(model.config, "max_position_embeddings"):
                    if int(max_seq_length) > 0:
                        model.config.max_position_embeddings = int(max_seq_length)

    try:
        if getattr(model, "generation_config", None) is not None:
            gc = model.generation_config
            if gc.pad_token_id is None:
                gc.pad_token_id = tokenizer.pad_token_id
            if gc.eos_token_id is None:
                gc.eos_token_id = tokenizer.eos_token_id
        if getattr(model, "config", None) is not None:
            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    return model, tokenizer


def initialize_model(cfg: Dict[str, Any]):
    provider = (cfg.get("provider") or "unsloth").lower()

    if provider in {"unsloth", "unsloth-fast"}:
        model, tokenizer = initialize_unsloth_model(cfg)
    elif provider in {"hf", "huggingface"}:
        model, tokenizer = initialize_hf_model(cfg)
    else:
        raise ValueError(f"Unsupported model.provider '{cfg.get('provider')}'.")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "provider": provider,
    }
def build_prompt(
    tokenizer,
    system_prompt: str | None,
    template: str,
    *,
    input_text: str,
    label_name: str,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })
    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": template.format(input_text=input_text, label_name=label_name),
        }],
    })
    return messages


def prepare_generation_cfg(defaults: Dict[str, Any], prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    generation_cfg = dict(defaults)
    generation_cfg.update(prompt_cfg.get("generation") or {})

    for key in GENERATION_PARAM_KEYS:
        if key in prompt_cfg:
            generation_cfg[key] = prompt_cfg[key]

    return generation_cfg


def resolve_prompt_selection(
    prompts_cfg: Dict[str, Dict[str, Any]],
    names: Sequence[str] | None,
) -> List[Tuple[str, Dict[str, Any]]]:
    if not prompts_cfg:
        raise ValueError("augmentation.prompts が設定されていません。")

    if not names:
        return list(prompts_cfg.items())

    missing = [name for name in names if name not in prompts_cfg]
    if missing:
        raise KeyError(
            "指定されたプロンプトが見つかりません: " + ", ".join(sorted(missing))
        )

    return [(name, prompts_cfg[name]) for name in names]


import torch
from typing import Any, Dict, Iterable, List, Sequence

@torch.no_grad()
def generate_batch_texts(
    model_bundle: Dict[str, Any],
    prompts: Sequence[List[Dict[str, Any]]],   # プロンプト配列
    generation_cfg: Dict[str, Any],
) -> List[str]:
    if not prompts:
        return []

    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]

    rendered_prompts = render_chat_prompts(tokenizer, prompts)

    if (
        getattr(tokenizer, "pad_token_id", None) is not None
        and tokenizer.pad_token_id == getattr(tokenizer, "eos_token_id", None)
    ):
        rendered_prompts = [rp + " " for rp in rendered_prompts]

    prev_padding_side = getattr(tokenizer, "padding_side", "right")
    prev_truncation_side = getattr(tokenizer, "truncation_side", "right")
    attention_mask = None
    try:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        encoded = tokenizer(
            rendered_prompts,
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
            add_special_tokens=False,
            return_tensors="pt",
        )
        model_device = get_model_device(model)
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded["attention_mask"].to(model_device)

        generation_kwargs = {
            "max_new_tokens": int(generation_cfg.get("max_new_tokens", 256)),
            "temperature": float(generation_cfg.get("temperature", 0.8)),
            "top_p": float(generation_cfg.get("top_p", 0.95)),
            "do_sample": bool(generation_cfg.get("do_sample", True)),
            "repetition_penalty": float(generation_cfg.get("repetition_penalty", 1.05)),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if "top_k" in generation_cfg:
            generation_kwargs["top_k"] = int(generation_cfg["top_k"])

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    finally:
        tokenizer.padding_side = prev_padding_side
        tokenizer.truncation_side = prev_truncation_side

    if attention_mask is None:
        return [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

    input_lengths = attention_mask.sum(dim=1).tolist()
    texts: List[str] = []
    for i, length in enumerate(input_lengths):
        gen_ids = outputs[i, int(length):]
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts
def append_identifier_to_path(path: Path, identifier: str) -> Path:
    if not identifier:
        return path
    suffix = path.suffix
    if suffix:
        return path.with_name(f"{path.stem}_{identifier}{suffix}")
    return path.parent / f"{path.name}_{identifier}"


def determine_output_path(
    prompt_name: str,
    prompt_cfg: Dict[str, Any],
    default_prefix: str,
    run_index: int,
    total_runs: int,
    *,
    format_values: Dict[str, Any] | None = None,
    multiple_datasets: bool = False,
) -> Path:
    format_values = dict(format_values or {})
    dataset_name = format_values.get("dataset_name")
    template = prompt_cfg.get("output_path_template")
    if template:
        used_dataset_placeholder = dataset_name is not None and "{dataset_name}" in template
        try:
            path_str = template.format(run_index=run_index, **format_values)
        except KeyError as err:
            raise KeyError(
                f"Prompt '{prompt_name}' �� output_path_template �� '{err.args[0]}' ���܂߂Ă��������B"
            ) from err
        output_path = Path(path_str)
        if multiple_datasets and dataset_name and not used_dataset_placeholder:
            output_path = append_identifier_to_path(output_path, dataset_name)
        return output_path

    base_path_str = prompt_cfg.get("output_path")
    if base_path_str:
        base_path = Path(base_path_str)
    else:
        base_path = Path(f"{default_prefix}_{prompt_name}.jsonl")

    if not base_path.suffix:
        base_path = Path(str(base_path) + ".jsonl")

    if multiple_datasets and dataset_name:
        base_path = append_identifier_to_path(base_path, dataset_name)

    if total_runs == 1:
        return base_path

    return append_identifier_to_path(base_path, f"run{run_index}")


def main(
    config_path: str = "augmentation.yaml",
    limit: int | None = None,
    prompt_names: Sequence[str] | None = None,
    list_prompts: bool = False,
) -> None:
    config = load_config(Path(config_path))
    aug_cfg = config.get("augmentation") or {}
    dataset_cfg = aug_cfg.get("dataset") or {}
    model_cfg = aug_cfg.get("model") or {}
    generation_defaults = aug_cfg.get("generation_defaults") or {}
    prompts_cfg = aug_cfg.get("prompts") or {}

    if list_prompts:
        if not prompts_cfg:
            print("[Info] augmentation.prompts が見つかりません")
            return
        print("[Info] Registered prompts:")
        for name, prompt in prompts_cfg.items():
            description = prompt.get("description")
            if description:
                print(f"  - {name}: {description}")
            else:
                print(f"  - {name}")
        return

    selected_prompts = resolve_prompt_selection(prompts_cfg, prompt_names)

    if limit is not None and limit <= 0:
        raise ValueError("limit は正の整数である必要があります")

    dataset_cache: Dict[Tuple[Path, int | None], Dataset] = {}

    
    def resolve_prompt_datasets(
        prompt_name: str,
        prompt_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        merged_dataset_cfg = dict(dataset_cfg)
        prompt_dataset_cfg = prompt_cfg.get("dataset") or {}
        merged_dataset_cfg.update(prompt_dataset_cfg)

        raw_paths: List[Any] = []
        paths_value = merged_dataset_cfg.get("input_paths")
        if paths_value:
            if isinstance(paths_value, (str, Path)):
                raw_paths = [paths_value]
            else:
                raw_paths = list(paths_value)
        else:
            dataset_path_value = merged_dataset_cfg.get("input_path")
            if not dataset_path_value:
                raise ValueError(
                    f"Prompt '{prompt_name}' �� dataset.input_path ������܂���"
                )
            raw_paths = [dataset_path_value]

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

    model_cache: Dict[str, Dict[str, Any]] = {}

    for prompt_name, prompt_cfg in selected_prompts:
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

        generation_cfg = prepare_generation_cfg(generation_defaults, prompt_cfg)

        user_template = prompt_cfg.get("user_prompt_template")
        if not user_template:
            raise ValueError(f"Prompt '{prompt_name}' �� user_prompt_template ������܂���")

        num_generations = int(generation_cfg.get("num_generations_per_sample", 1))
        if num_generations <= 0:
            raise ValueError(
                f"Prompt '{prompt_name}' has invalid num_generations_per_sample={num_generations}."
            )

        batch_size_raw = generation_cfg.get("batch_size", 1)
        try:
            batch_size = max(1, int(batch_size_raw))
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Prompt '{prompt_name}' �� batch_size ���s���ł�"
            ) from err

        dataset_entries = resolve_prompt_datasets(prompt_name, prompt_cfg)
        if not dataset_entries:
            continue

        multiple_datasets_for_prompt = len(dataset_entries) > 1
        source_name = effective_model_cfg.get("name") or effective_model_cfg.get("model_name")

        for dataset_entry in dataset_entries:
            subset_dataset = dataset_entry["dataset"]
            prompt_dataset_cfg = dataset_entry["dataset_cfg"]
            dataset_path = dataset_entry["dataset_path"]
            dataset_name = dataset_entry["dataset_name"]
            dataset_index = dataset_entry["dataset_index"]

            text_col = prompt_dataset_cfg.get("text_column")
            if text_col is None:
                text_col = dataset_cfg.get("text_column") if dataset_cfg else None
            text_col = text_col or "sentence"

            label_col = prompt_dataset_cfg.get("label_column")
            if label_col is None:
                label_col = dataset_cfg.get("label_column") if dataset_cfg else None
            label_col = label_col or "label"

            label_names_value = prompt_dataset_cfg.get("label_names")
            if label_names_value is None:
                label_names_value = dataset_cfg.get("label_names") if dataset_cfg else None
            label_names = list(label_names_value) if label_names_value else []

            default_output_prefix_value = prompt_dataset_cfg.get("default_output_prefix")
            if default_output_prefix_value is None:
                default_output_prefix_value = dataset_cfg.get("default_output_prefix") if dataset_cfg else None
            default_output_prefix = default_output_prefix_value or "data/marc_ja_augmented"

            if text_col not in subset_dataset.column_names or label_col not in subset_dataset.column_names:
                raise KeyError(
                    f"Prompt '{prompt_name}' �̃f�[�^�� '{text_col}' �܂��� '{label_col}' ��������܂��� "
                    f"(path={prompt_dataset_cfg.get('input_path')})"
                )

            dataset_length = len(subset_dataset)
            if dataset_length == 0:
                print(
                    f"[Warning] Dataset '{prompt_dataset_cfg.get('input_path')}' is empty. Prompt '{prompt_name}' �̓X�L�b�v���܂��B"
                )
                continue

            if label_names:
                try:
                    max_label_id = max(int(v) for v in subset_dataset[label_col])
                except ValueError as err:
                    raise ValueError(
                        f"Prompt '{prompt_name}' �� '{label_col}' �񂪐������x���ł͂���܂���"
                    ) from err
                if max_label_id >= len(label_names):
                    raise ValueError(
                        f"Prompt '{prompt_name}' : label_names �̐���胉�x��ID���傫���l���܂܂�܂�"
                    )

            per_pass_capacity = dataset_length * num_generations

            multiplier_raw = prompt_cfg.get("generation_multiplier")
            if multiplier_raw is not None:
                try:
                    multiplier = float(multiplier_raw)
                except (TypeError, ValueError) as err:
                    raise ValueError(
                        f"Prompt '{prompt_name}' �� generation_multiplier ���s���ł�"
                    ) from err
                if multiplier <= 0:
                    raise ValueError(
                        f"Prompt '{prompt_name}' �� generation_multiplier �͐��̐��ł���K�v������܂�"
                    )
                total_required = int(math.ceil(dataset_length * multiplier))
            else:
                total_required = per_pass_capacity

            if total_required <= 0:
                print(f"[Warning] Prompt '{prompt_name}' dataset '{dataset_name}' �̐����v���� 0 ���ł��B�X�L�b�v���܂��B")
                continue

            runs_needed = 1 if per_pass_capacity == 0 else max(1, math.ceil(total_required / per_pass_capacity))

            prompt_seed = generation_cfg.get("seed")
            if prompt_seed is None:
                prompt_seed = aug_cfg.get("seed")

            total_generated = 0

            for run_index in range(1, runs_needed + 1):
                remaining = total_required - total_generated
                if remaining <= 0:
                    break

                run_seed = prompt_seed + run_index - 1 if prompt_seed is not None else None
                if run_seed is not None:
                    set_seed(run_seed)
                    run_dataset = subset_dataset.shuffle(seed=run_seed)
                else:
                    run_dataset = subset_dataset.shuffle(seed=random.randint(0, 1_000_000)) if runs_needed > 1 else subset_dataset

                run_records: List[Dict[str, Any]] = []
                pending_prompts: List[List[Dict[str, Any]]] = []
                pending_meta: List[Dict[str, Any]] = []

                def flush_pending() -> None:
                    nonlocal pending_prompts, pending_meta, total_generated
                    if not pending_prompts:
                        return
                    texts = generate_batch_texts(model_bundle, pending_prompts, generation_cfg)
                    for text_output, meta in zip(texts, pending_meta):
                        if total_generated >= total_required:
                            break
                        if not text_output:
                            continue
                        run_records.append(
                            {
                                text_col: text_output,
                                label_col: meta["label_id"],
                                "source": source_name,
                                "prompt_name": prompt_name,
                                "reference_text": meta["reference_text"],
                            }
                        )
                        total_generated += 1
                    pending_prompts = []
                    pending_meta = []

                progress_desc = f"Generating ({prompt_name}, run {run_index})"
                iterable: Iterable[Dict[str, Any]] = run_dataset

                for example in tqdm(iterable, desc=progress_desc, total=dataset_length):
                    if total_generated >= total_required:
                        break

                    label_id = example[label_col]
                    label_name = (
                        label_names[int(label_id)]
                        if label_names
                        else str(label_id)
                    )

                    messages = build_prompt(
                        tokenizer,
                        prompt_cfg.get("system_prompt", "") or None,
                        user_template,
                        input_text=example[text_col],
                        label_name=label_name,
                    )

                    for _ in range(num_generations):
                        if total_generated >= total_required:
                            break
                        remaining_after_queue = total_required - total_generated - len(pending_prompts)
                        if remaining_after_queue <= 0:
                            break
                        pending_prompts.append(messages)
                        pending_meta.append(
                            {
                                "label_id": label_id,
                                "reference_text": example[text_col],
                            }
                        )
                        if len(pending_prompts) >= batch_size:
                            flush_pending()

                    if total_generated >= total_required:
                        break

                flush_pending()

                if not run_records:
                    print(
                        f"[Warning] Prompt '{prompt_name}' dataset '{dataset_name}' run {run_index} �ŗL���ȃT���v������������܂���ł����B"
                    )
                    continue

                format_values = {
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "dataset_index": dataset_index,
                }
                output_path = determine_output_path(
                    prompt_name,
                    prompt_cfg,
                    default_output_prefix,
                    run_index,
                    runs_needed,
                    format_values=format_values,
                    multiple_datasets=multiple_datasets_for_prompt,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as f:
                    for record in run_records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(
                    f"[Info] Prompt '{prompt_name}' dataset '{dataset_name}' run {run_index}/{runs_needed} saved {len(run_records)} samples to {output_path} "
                    f"(total {total_generated}/{total_required})."
                )

            if total_generated == 0:
                print(
                    f"[Warning] Prompt '{prompt_name}' dataset '{dataset_name}' ��1���������ł��܂���ł����B"
                )
            elif total_generated < total_required:
                print(
                    f"[Warning] Prompt '{prompt_name}' dataset '{dataset_name}' �� {total_generated}/{total_required} ���̂ݐ�������܂����B"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented data with UnsLoTh LLM prompts.")
    parser.add_argument("--config", default="augmentation.yaml", help="Path to the augmentation config file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of subset examples processed.")
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Specify prompt name(s) to run. Can be used multiple times.",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt names and exit.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        limit=args.limit,
        prompt_names=args.prompts,
        list_prompts=args.list_prompts,
    )
