import argparse
import asyncio
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import uuid4

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from augment import (
    build_prompt,
    determine_output_path,
    infer_chat_template_name,
    load_config,
    load_subset_dataset,
    prepare_generation_cfg,
    resolve_prompt_selection,
    set_seed,
)

try:
    from datasets import Dataset
except ImportError as err:  # pragma: no cover - datasets is a required dependency
    raise ImportError(
        "datasets パッケージが見つかりません。pip install datasets でインストールしてください。"
    ) from err

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdmは任意依存
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


ENGINE_REQUEST_OPTION_DEFAULTS = {
    "poll_interval": 0.01,
}


def render_chat_messages(
    messages: Sequence[Dict[str, Any]],
    tokenizer: Any | None = None,
) -> str:
    """Render chat messages, preferring tokenizer.apply_chat_template when available."""
    normalized: List[Dict[str, str]] = []
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
        normalized.append({"role": role, "text": text})

    apply_chat_template = (
        getattr(tokenizer, "apply_chat_template", None)
        if tokenizer is not None
        else None
    )

    if callable(apply_chat_template):
        hf_messages = [
            {"role": entry["role"], "content": entry["text"]}
            for entry in normalized
            if entry["text"]
        ]
        if hf_messages:
            try:
                rendered = apply_chat_template(
                    hf_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception as err:
                print(
                    f"[Warning] Failed to render chat with tokenizer template; falling back to manual formatting: {err}"
                )

    parts: List[str] = []
    for entry in normalized:
        text = entry["text"]
        if not text:
            continue
        role = entry["role"]
        if role == "system":
            parts.append(text)
        elif role == "assistant":
            parts.append(f"Assistant:\n{text}")
        elif role == "user":
            parts.append(text)
        else:
            parts.append(f"{role.capitalize()}:\n{text}")
    return ("\n\n".join(parts).strip() + "\n\n") if parts else ""


def build_sampling_params(config: Dict[str, Any]) -> SamplingParams:
    params: Dict[str, Any] = {}
    max_tokens = config.get("max_new_tokens")
    if max_tokens is None:
        max_tokens = 256
    params["max_tokens"] = int(max_tokens)

    do_sample = config.get("do_sample", True)
    temperature = float(config.get("temperature", 1.0))
    top_p = float(config.get("top_p", 0.95))
    top_k = config.get("top_k")

    if not do_sample:
        params["temperature"] = 0.0
        params["top_p"] = 1.0
        if top_k is not None:
            params["top_k"] = int(top_k)
    else:
        params["temperature"] = temperature
        params["top_p"] = top_p
        if top_k is not None:
            params["top_k"] = int(top_k)

    repetition_penalty = config.get("repetition_penalty")
    if repetition_penalty is not None:
        params["repetition_penalty"] = float(repetition_penalty)

    seed = config.get("seed")
    if seed is not None:
        params["seed"] = int(seed)

    return SamplingParams(**params)


def normalize_request_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    options = dict(ENGINE_REQUEST_OPTION_DEFAULTS)
    options.update(cfg)
    if "max_concurrent_requests" in options and options["max_concurrent_requests"] is not None:
        options["max_concurrent_requests"] = max(1, int(options["max_concurrent_requests"]))
    options["poll_interval"] = float(options.get("poll_interval", ENGINE_REQUEST_OPTION_DEFAULTS["poll_interval"]))
    return options


def make_engine_cache_key(cfg: Dict[str, Any]) -> str:
    normalized = {
        "name": cfg.get("name") or cfg.get("model") or cfg.get("model_name"),
        "engine_args": cfg.get("engine_args"),
        "chat_template": cfg.get("chat_template"),
        "tokenizer": cfg.get("tokenizer"),
    }
    return json.dumps(normalized, sort_keys=True, default=str)


def create_engine_bundle(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    engine_args_cfg = dict(model_cfg.get("engine_args") or {})
    model_name = (
        model_cfg.get("name")
        or model_cfg.get("model_name")
        or engine_args_cfg.get("model")
    )
    if not model_name:
        raise ValueError("model.name または model.engine_args.model を指定してください。")
    engine_args_cfg.setdefault("model", model_name)

    tokenizer_cfg = model_cfg.get("tokenizer")
    if isinstance(tokenizer_cfg, dict):
        tokenizer_name = tokenizer_cfg.get("name") or tokenizer_cfg.get("path")
        if tokenizer_name:
            engine_args_cfg.setdefault("tokenizer", tokenizer_name)
    elif isinstance(tokenizer_cfg, str):
        engine_args_cfg.setdefault("tokenizer", tokenizer_cfg)

    tokenizer_name = model_cfg.get("tokenizer_name")
    if tokenizer_name and "tokenizer" not in engine_args_cfg:
        engine_args_cfg["tokenizer"] = tokenizer_name

    engine_args = AsyncEngineArgs(**engine_args_cfg)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    chat_template = model_cfg.get("chat_template")
    if not chat_template:
        inferred = infer_chat_template_name(model_name)
        chat_template = inferred

    request_options = normalize_request_options(model_cfg.get("request_options") or {})

    tokenizer_obj: Any | None = None
    try:
        tokenizer_obj = engine.get_tokenizer()
        if tokenizer_obj is not None and chat_template and hasattr(tokenizer_obj, "chat_template"):
            current_template = getattr(tokenizer_obj, "chat_template", None)
            if not current_template:
                setattr(tokenizer_obj, "chat_template", chat_template)
    except Exception as err:
        print(f"[Warning] Failed to fetch tokenizer from AsyncLLMEngine: {err}")
        tokenizer_obj = None

    return {
        "engine": engine,
        "engine_args": engine_args,
        "chat_template": chat_template,
        "request_options": request_options,
        "tokenizer": tokenizer_obj,
    }


async def shutdown_engine(engine: AsyncLLMEngine) -> None:
    if hasattr(engine, "shutdown"):
        result = engine.shutdown()  # type: ignore[attr-defined]
        if asyncio.iscoroutine(result):
            await result
    elif hasattr(engine, "stop"):
        result = engine.stop()  # type: ignore[attr-defined]
        if asyncio.iscoroutine(result):
            await result


async def generate_batch_async(
    engine: AsyncLLMEngine,
    prompts: Sequence[str],
    sampling_params: SamplingParams,
    *,
    max_concurrent: int,
    poll_interval: float,
) -> List[str]:
    if not prompts:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[str] = [""] * len(prompts)

    async def worker(index: int, prompt: str) -> None:
        request_id = f"req-{index}-{uuid4().hex}"
        await semaphore.acquire()
        try:
            try:
                add_result = await engine.add_request(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=sampling_params,
                )
            except TypeError as err:
                if 'sampling_params' in str(err) or 'unexpected keyword' in str(err):
                    add_result = await engine.add_request(request_id, prompt, sampling_params)
                else:
                    raise

            collector = add_result if hasattr(add_result, 'get') else None
            rid = None if collector is not None else add_result
            if rid is None:
                rid = request_id

            if collector is not None:
                while True:
                    result = await collector.get()
                    outputs = getattr(result, 'outputs', None)
                    if outputs:
                        results[index] = (outputs[0].text or "").strip()
                    if getattr(result, 'finished', True):
                        break
            else:
                while True:
                    result = await engine.get_result(rid)
                    if result is not None:
                        outputs = getattr(result, 'outputs', None)
                        if outputs:
                            results[index] = (outputs[0].text or "").strip()
                        if getattr(result, 'finished', True):
                            break
                    await asyncio.sleep(poll_interval)
        finally:
            semaphore.release()

    tasks = [asyncio.create_task(worker(i, prompt)) for i, prompt in enumerate(prompts)]
    await asyncio.gather(*tasks)
    return results


async def async_main(
    *,
    config_path: str,
    limit: int | None,
    prompt_names: Sequence[str] | None,
    list_prompts: bool,
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
    engine_cache: Dict[str, Dict[str, Any]] = {}

    async def get_engine_bundle(effective_model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = make_engine_cache_key(effective_model_cfg)
        if cache_key not in engine_cache:
            engine_cache[cache_key] = create_engine_bundle(effective_model_cfg)
        return engine_cache[cache_key]

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
                    f"Prompt '{prompt_name}' の dataset.input_path が指定されていません"
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

    try:
        for prompt_name, prompt_cfg in selected_prompts:
            effective_model_cfg = dict(model_cfg)
            prompt_model_cfg = prompt_cfg.get("model") or {}
            for key, value in prompt_model_cfg.items():
                if key == "engine_args" and key in effective_model_cfg:
                    base_engine_args = dict(effective_model_cfg.get("engine_args") or {})
                    base_engine_args.update(value)
                    effective_model_cfg["engine_args"] = base_engine_args
                else:
                    effective_model_cfg[key] = value

            engine_bundle = await get_engine_bundle(effective_model_cfg)
            engine = engine_bundle["engine"]
            request_options = engine_bundle["request_options"]
            tokenizer = engine_bundle.get("tokenizer")

            generation_cfg = prepare_generation_cfg(generation_defaults, prompt_cfg)
            sampling_params = build_sampling_params(generation_cfg)

            user_template = prompt_cfg.get("user_prompt_template")
            if not user_template:
                raise ValueError(f"Prompt '{prompt_name}' に user_prompt_template が存在しません")

            num_generations = int(generation_cfg.get("num_generations_per_sample", 1))
            if num_generations <= 0:
                raise ValueError(
                    f"Prompt '{prompt_name}' の num_generations_per_sample が不正です: {num_generations}"
                )

            batch_size_raw = generation_cfg.get("batch_size", 1)
            try:
                batch_size = max(1, int(batch_size_raw))
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Prompt '{prompt_name}' の batch_size を整数に変換できません"
                ) from err

            dataset_entries = resolve_prompt_datasets(prompt_name, prompt_cfg)
            if not dataset_entries:
                continue

            multiple_datasets_for_prompt = len(dataset_entries) > 1
            source_name = (
                effective_model_cfg.get("name")
                or effective_model_cfg.get("model_name")
                or effective_model_cfg.get("engine_args", {}).get("model")
            )

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
                        f"Prompt '{prompt_name}' 用のデータセット {dataset_path} に列 '{text_col}' または '{label_col}' が存在しません"
                    )

                dataset_length = len(subset_dataset)
                if dataset_length == 0:
                    print(
                        f"[Warning] Dataset '{dataset_path}' は空のため Prompt '{prompt_name}' をスキップします"
                    )
                    continue

                if label_names:
                    try:
                        max_label_id = max(int(v) for v in subset_dataset[label_col])
                    except ValueError as err:
                        raise ValueError(
                            f"Prompt '{prompt_name}' の '{label_col}' に整数以外の値が含まれています"
                        ) from err
                    if max_label_id >= len(label_names):
                        raise ValueError(
                            f"Prompt '{prompt_name}' の label_names の長さが不足しています"
                        )

                per_pass_capacity = dataset_length * num_generations

                multiplier_raw = prompt_cfg.get("generation_multiplier")
                if multiplier_raw is not None:
                    try:
                        multiplier = float(multiplier_raw)
                    except (TypeError, ValueError) as err:
                        raise ValueError(
                            f"Prompt '{prompt_name}' の generation_multiplier を数値に変換できません"
                        ) from err
                    if multiplier <= 0:
                        raise ValueError(
                            f"Prompt '{prompt_name}' の generation_multiplier は正の値である必要があります"
                        )
                    total_required = int(math.ceil(dataset_length * multiplier))
                else:
                    total_required = per_pass_capacity

                if total_required <= 0:
                    print(
                        f"[Warning] Prompt '{prompt_name}' dataset '{dataset_name}' の生成数が 0 のためスキップします"
                    )
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
                        run_dataset = (
                            subset_dataset.shuffle(seed=random.randint(0, 1_000_000))
                            if runs_needed > 1
                            else subset_dataset
                        )

                    run_records: List[Dict[str, Any]] = []
                    pending_prompts: List[str] = []
                    pending_meta: List[Dict[str, Any]] = []

                    async def flush_pending() -> None:
                        nonlocal pending_prompts, pending_meta, total_generated
                        if not pending_prompts:
                            return
                        outputs = await generate_batch_async(
                            engine,
                            pending_prompts,
                            sampling_params,
                            max_concurrent=request_options.get("max_concurrent_requests", batch_size),
                            poll_interval=request_options["poll_interval"],
                        )
                        for output_text, meta in zip(outputs, pending_meta):
                            if total_generated >= total_required:
                                break
                            if not output_text:
                                continue
                            run_records.append(
                                {
                                    text_col: output_text,
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
                            None,
                            prompt_cfg.get("system_prompt", "") or None,
                            user_template,
                            input_text=example[text_col],
                            label_name=label_name,
                        )
                        rendered_prompt = render_chat_messages(messages, tokenizer=tokenizer)

                        for _ in range(num_generations):
                            if total_generated >= total_required:
                                break
                            remaining_after_queue = total_required - total_generated - len(pending_prompts)
                            if remaining_after_queue <= 0:
                                break
                            pending_prompts.append(rendered_prompt)
                            pending_meta.append(
                                {
                                    "label_id": label_id,
                                    "reference_text": example[text_col],
                                }
                            )
                            if len(pending_prompts) >= batch_size:
                                await flush_pending()

                        if total_generated >= total_required:
                            break

                    await flush_pending()

                    if not run_records:
                        print(
                            f"[Warning] Prompt '{prompt_name}' dataset '{dataset_name}' run {run_index} で有効な生成がありませんでした"
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
                        f"[Info] Prompt '{prompt_name}' dataset '{dataset_name}' run {run_index}/{runs_needed} で {len(run_records)} 件を {output_path} に保存 (累計 {total_generated}/{total_required})"
                    )

                if total_generated == 0:
                    print(
                        f"[Warning] Prompt '{prompt_name}' dataset '{dataset_name}' で生成されたサンプルがありません"
                    )
                elif total_generated < total_required:
                    print(
                        f"[Warning] Prompt '{prompt_name}' dataset '{dataset_name}' は {total_generated}/{total_required} 件のみ生成されました"
                    )
    finally:
        await asyncio.gather(
            *(shutdown_engine(bundle["engine"]) for bundle in engine_cache.values()),
            return_exceptions=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate augmented data with vLLM AsyncLLMEngine without standing up a server.",
    )
    parser.add_argument(
        "--config",
        default="augmentation_vllm.yaml",
        help="Path to the augmentation config file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of subset examples processed.",
    )
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
    asyncio.run(
        async_main(
            config_path=args.config,
            limit=args.limit,
            prompt_names=args.prompts,
            list_prompts=args.list_prompts,
        )
    )


if __name__ == "__main__":
    main()
