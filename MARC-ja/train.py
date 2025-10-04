import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate


def resolve_config_path(config_path: str | Path) -> Path:
    """Resolve config path from CWD or module directory with helpful errors."""

    raw_path = Path(config_path).expanduser()
    module_dir = Path(__file__).resolve().parent

    search_candidates: list[Path] = [raw_path]

    if not raw_path.is_absolute():
        search_candidates.extend(
            [
                Path.cwd() / raw_path,
                module_dir / raw_path,
            ]
        )

    # Remove duplicates while preserving order for clearer error messages.
    seen: set[Path] = set()
    candidates: list[Path] = []
    for candidate in search_candidates:
        resolved_candidate = candidate
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        candidates.append(resolved_candidate)
        if resolved_candidate.is_file():
            return resolved_candidate.resolve()

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find config file '{config_path}'. Tried:\n{searched}"
    )


def load_tokenizer_with_fallback(model_cfg: dict):
    tokenizer_cfg = model_cfg.get("tokenizer", {})
    use_fast = tokenizer_cfg.get("use_fast", True)

    tokenizer_kwargs = dict(tokenizer_cfg.get("kwargs", {}))
    for key in (
        "word_tokenizer_type",
        "do_word_tokenize",
        "do_subword_tokenize",
        "mecab_kwargs",
        "sudachi_kwargs",
        "jumanpp_kwargs",
    ):
        if key in tokenizer_cfg and key not in tokenizer_kwargs:
            tokenizer_kwargs[key] = tokenizer_cfg[key]

    fallback_to_basic = tokenizer_cfg.get("fallback_to_basic_if_unavailable", False)

    try:
        return AutoTokenizer.from_pretrained(
            model_cfg["name"], use_fast=use_fast, **tokenizer_kwargs
        )
    except ModuleNotFoundError as err:
        message = str(err).lower()
        if fallback_to_basic and "fugashi" in message:
            print(
                "[Warning] Required library 'fugashi' is not installed. "
                "Falling back to the basic tokenizer. Install `pip install fugashi ipadic` "
                "for better tokenization quality."
            )
            basic_kwargs = dict(tokenizer_kwargs)
            basic_kwargs["word_tokenizer_type"] = "basic"
            return AutoTokenizer.from_pretrained(
                model_cfg["name"], use_fast=use_fast, **basic_kwargs
            )
        raise


def main(config_path: str | Path = "config.yaml"):
    # ============ 1) 險ｭ螳壹ヵ繧｡繧､繝ｫ繧定ｪｭ縺ｿ霎ｼ縺ｿ ============
    config_file = resolve_config_path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config.get("data", {})
    wandb_cfg = config.get("wandb", {})

    # ============ 2) 繝・・繧ｿ繧ｻ繝・ヨ隱ｭ縺ｿ霎ｼ縺ｿ ============
    ds = load_dataset("shunk031/JGLUE", name="MARC-ja", trust_remote_code=True)
    text_col, label_col = "sentence", "label"

    report_to = train_cfg.get("report_to", "none")
    wandb_run = None
    wandb_module = None
    if report_to == "wandb":
        try:
            import wandb as wandb_module
            from wandb.errors import Error as WandbError
        except ImportError as err:
            raise ImportError(
                "report_to is set to 'wandb' but the wandb package is not installed. "
                "Install it with `pip install wandb`."
            ) from err

        wandb_kwargs = {}
        for key in ("project", "entity", "run_name", "notes"):
            value = wandb_cfg.get(key)
            if value:
                arg_key = "name" if key == "run_name" else key
                wandb_kwargs[arg_key] = value

        tags = wandb_cfg.get("tags")
        if tags:
            wandb_kwargs["tags"] = tags

        try:
            wandb_run = wandb_module.init(**wandb_kwargs)
        except WandbError as err:
            print(
                "[Warning] Failed to initialise Weights & Biases logging. "
                f"Falling back to 'none'.\nReason: {err}"
            )
            wandb_run = None
            wandb_module = None
            report_to = "none"
        else:
            wandb_run.config.update(
                {
                    "model": model_cfg,
                    "training": train_cfg,
                    "data": data_cfg,
                },
                allow_val_change=True,
            )

    subset_path_value = data_cfg.get("train_subset_path")
    subset_enabled = data_cfg.get("train_subset_enabled", False)

    if subset_enabled:
        if not subset_path_value:
            raise ValueError("data.train_subset_enabled is true but data.train_subset_path is not set.")
        subset_path = Path(subset_path_value)
        if not subset_path.exists():
            raise FileNotFoundError(
                f"Subset file not found at {subset_path}. Run make_sampling_data.py to generate it."
            )
        subset_dataset = Dataset.from_json(str(subset_path))
        ds["train"] = subset_dataset
        print(
            f"[Info] Training will use the saved subset ({len(subset_dataset)} samples) from {subset_path.resolve()}"
        )
    else:
        print("[Info] Training will use the full training dataset.")


    mix_cfg = data_cfg.get("augmented_mix") or {}
    if mix_cfg.get("enabled"):
        augmented_paths = mix_cfg.get("augmented_paths") or []
        if not augmented_paths:
            raise ValueError("data.augmented_mix.enabled が true の場合、augmented_paths を指定してください。")

        augmented_datasets = []
        for path_str in augmented_paths:
            augmented_path = Path(path_str)
            if not augmented_path.exists():
                raise FileNotFoundError(f"Augmented dataset not found at {augmented_path}.")
            augmented_ds = Dataset.from_json(str(augmented_path))
            if text_col not in augmented_ds.column_names or label_col not in augmented_ds.column_names:
                raise KeyError(
                    f"Augmented dataset at {augmented_path} is missing required columns '{text_col}' and '{label_col}'."
                )
            augmented_datasets.append(augmented_ds)
            try:
                resolved_aug_path = augmented_path.resolve()
            except RuntimeError:
                resolved_aug_path = augmented_path
            print(
                f"[Info] Added augmented dataset ({len(augmented_ds)} samples) from {resolved_aug_path}"
            )

        if not augmented_datasets:
            raise ValueError("augmented_paths に有効なファイルがありません。")

        if len(augmented_datasets) == 1:
            augmented_combined = augmented_datasets[0]
        else:
            augmented_combined = concatenate_datasets(augmented_datasets)

        original_fraction = float(mix_cfg.get("original_fraction", 1.0))
        if original_fraction < 0 or original_fraction > 1:
            raise ValueError("data.augmented_mix.original_fraction must be between 0 and 1.")

        train_dataset = ds["train"]
        sample_seed = mix_cfg.get("seed", train_cfg.get("seed"))

        if original_fraction <= 0:
            sample_size = 0
            sampled_original = None
        else:
            sample_size = max(1, int(len(train_dataset) * original_fraction))
            if sample_size < len(train_dataset):
                sampled_original = train_dataset.shuffle(seed=sample_seed).select(range(sample_size))
            else:
                sampled_original = train_dataset

        datasets_to_concat = [augmented_combined] if sampled_original is None else [sampled_original, augmented_combined]
        combined_train = concatenate_datasets(datasets_to_concat)

        if mix_cfg.get("shuffle_after_concat", True):
            combined_train = combined_train.shuffle(seed=sample_seed)

        ds["train"] = combined_train
        if sample_size > 0:
            print(
                f"[Info] Training will mix {sample_size} original samples ({original_fraction:.2%}) with {len(augmented_combined)} augmented samples."
            )
        else:
            print(
                f"[Info] Training will use only augmented samples ({len(augmented_combined)} records)."
            )

    label_feature = ds["train"].features[label_col]
    dataset_num_labels = getattr(label_feature, "num_classes", None)
    if dataset_num_labels is None:
        dataset_num_labels = len(set(ds["train"][label_col]))

    observed_labels = set(int(v) for v in ds["train"][label_col])
    if "validation" in ds:
        observed_labels.update(int(v) for v in ds["validation"][label_col])
    observed_num_labels = len(observed_labels)
    if observed_num_labels and dataset_num_labels and observed_num_labels < dataset_num_labels:
        print(
            f"[Info] Dataset metadata reports {dataset_num_labels} labels but only "
            f"{observed_num_labels} are present. Using observed label count."
        )
        dataset_num_labels = observed_num_labels

    config_num_labels = model_cfg.get("num_labels")
    if config_num_labels is not None and config_num_labels != dataset_num_labels:
        print(
            f"[Warning] config num_labels={config_num_labels} does not match dataset "
            f"num_labels={dataset_num_labels}. Using dataset value."
        )

    num_labels = dataset_num_labels

    # ============ 3) 繝医・繧ｯ繝翫う繧ｶ ============
    tokenizer = load_tokenizer_with_fallback(model_cfg)

    def preprocess(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=model_cfg["max_len"]
        )

    ds_tokenized = ds.map(preprocess, batched=True, remove_columns=[text_col])

    # ============ 4) 繝｢繝・Ν ============
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=num_labels
    )

    if wandb_run is not None and wandb_cfg.get("watch_model", False):
        wandb_module.watch(model)

    disable_tqdm = bool(train_cfg.get("disable_tqdm", True))
    log_level = train_cfg.get("log_level", "error")
    log_level_replica = train_cfg.get("log_level_replica", log_level)

    # ============ 5) Collator & Metrics ============
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "f1_macro": f1_macro}

    # ============ 6) mixed precision 蟇ｾ蠢・============
    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_bf16 = train_cfg.get("bf16", False) and supports_bf16
    use_fp16 = train_cfg.get("fp16", False) and torch.cuda.is_available() and not use_bf16

    # ============ 7) TrainingArguments ============
    if isinstance(report_to, str):
        if report_to.lower() == "none":
            report_to_arg = []
        else:
            report_to_arg = [report_to]
    elif isinstance(report_to, (list, tuple)):
        report_to_arg = list(report_to)
    else:
        report_to_arg = None

    training_args = TrainingArguments(
        output_dir=str(train_cfg["output_dir"]),
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        learning_rate=float(train_cfg["lr"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        per_device_train_batch_size=int(train_cfg["train_bs"]),
        per_device_eval_batch_size=int(train_cfg["eval_bs"]),
        num_train_epochs=float(train_cfg["epochs"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=int(train_cfg["save_total_limit"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        seed=int(train_cfg["seed"]),
        report_to=report_to_arg,
        bf16=use_bf16,
        fp16=use_fp16,
        disable_tqdm=disable_tqdm,
        log_level=log_level,
        log_level_replica=log_level_replica,
    )

    # ============ 8) Trainer ============
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ============ 9) Train & Eval ============
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    best_metric = trainer.state.best_metric
    best_checkpoint = trainer.state.best_model_checkpoint

    best_f1_macro = None
    for record in trainer.state.log_history:
        value = record.get("eval_f1_macro")
        if value is not None:
            if best_f1_macro is None or value > best_f1_macro:
                best_f1_macro = float(value)

    if wandb_run is not None:
        log_payload: dict[str, float] = {}
        if "eval_accuracy" in metrics:
            log_payload["eval/accuracy"] = metrics["eval_accuracy"]
        if "eval_f1_macro" in metrics:
            log_payload["eval/f1_macro"] = metrics["eval_f1_macro"]
        if best_f1_macro is not None:
            log_payload["eval/best_f1_macro"] = best_f1_macro
        if best_metric is not None:
            log_payload["eval/best_accuracy"] = float(best_metric)
        if log_payload:
            wandb_run.log(log_payload)
        if best_metric is not None:
            wandb_run.summary["best_accuracy"] = float(best_metric)
        if best_f1_macro is not None:
            wandb_run.summary["best_f1_macro"] = best_f1_macro
        if best_checkpoint is not None:
            wandb_run.summary["best_model_checkpoint"] = best_checkpoint


    # 菫晏ｭ・    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier on MARC-ja dataset")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file (relative to CWD or module directory).",
    )
    cli_args = parser.parse_args()
    main(config_path=cli_args.config)

