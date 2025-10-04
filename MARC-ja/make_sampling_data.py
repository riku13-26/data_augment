import argparse
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset


def resolve_config_path(config_path: str | Path) -> Path:
    raw_path = Path(config_path).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(Path.cwd() / raw_path)
        module_dir = Path(__file__).resolve().parent
        candidates.append(module_dir / raw_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find config file '{config_path}'. Tried:\n{searched}"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
    config_path: str = "config.yaml",
    ratio: float | None = None,
    seed: int | None = None,
    output_path: str | None = None,
    dataset_id: str | None = None,
    dataset_config: str | None = None,
    overwrite: bool = False,
) -> None:
    config_file = resolve_config_path(config_path)
    config = load_yaml(config_file)

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    if ratio is None:
        ratio = data_cfg.get("train_subset_ratio")
    if ratio is None:
        raise ValueError(
            "Sampling ratio is not specified. Provide --ratio or set data.train_subset_ratio in the config."
        )
    ratio = float(ratio)
    if not 0 < ratio <= 1:
        raise ValueError("Sampling ratio must be within (0, 1].")

    if seed is None:
        seed = data_cfg.get("train_subset_seed")
    if seed is None:
        seed = training_cfg.get("seed")
    if seed is None:
        raise ValueError(
            "Seed is not specified. Provide --seed or set data.train_subset_seed/training.seed in the config."
        )
    seed = int(seed)

    if output_path is None:
        output_path = data_cfg.get("train_subset_path")
    if output_path is None:
        raise ValueError(
            "Output path is not specified. Provide --output or set data.train_subset_path in the config."
        )
    output_path = str(output_path)

    dataset_id = dataset_id or "shunk031/JGLUE"
    dataset_config = dataset_config or "MARC-ja"

    dataset = load_dataset(dataset_id, name=dataset_config, trust_remote_code=True)
    train_dataset = dataset["train"]

    sample_size = max(1, int(len(train_dataset) * ratio))
    subset = train_dataset.shuffle(seed=seed).select(range(sample_size))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{output}' already exists. Use --overwrite to replace it."
        )

    subset.to_json(str(output))

    print(
        f"Saved {sample_size} samples ({ratio:.2%}) to {output} "
        f"using seed={seed}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample a subset of the MARC-ja training data and store it locally."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--ratio", type=float, help="Fraction of the training split to sample.")
    parser.add_argument("--seed", type=int, help="Seed for deterministic shuffling.")
    parser.add_argument("--output", help="Destination JSONL path for the sampled subset.")
    parser.add_argument("--dataset", dest="dataset_id", help="Dataset identifier (default: shunk031/JGLUE).")
    parser.add_argument(
        "--dataset-config",
        dest="dataset_config",
        help="Dataset configuration name (default: MARC-ja).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it exists.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        ratio=args.ratio,
        seed=args.seed,
        output_path=args.output,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        overwrite=args.overwrite,
    )
