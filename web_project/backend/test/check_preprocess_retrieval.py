from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    if exc.name == "pandas":
        raise SystemExit(
            "Missing dependency: pandas\n"
            "Install it in the current environment first, for example:\n"
            "  pip install pandas==2.0.3"
        ) from exc
    raise


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed_data" / "web_project"
OUTPUT_DIR = DATA_DIR / "csv_exports"
PKL_FILES = (
    "train_eval_sample_final.pkl",
    "feature_dict.pkl",
    "vocab_dict.pkl",
)


def load_pickle(path: Path):
    with path.open("rb") as file:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="numpy.core.numeric is deprecated",
                category=DeprecationWarning,
            )
            return pickle.load(file)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def normalize_split_data(split_data: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in split_data.items():
        ndim = getattr(value, "ndim", None)
        if ndim == 1:
            normalized[key] = value
            continue
        if ndim and ndim > 1:
            normalized[key] = [json.dumps(row.tolist(), ensure_ascii=False) for row in value]
            continue
        normalized[key] = value
    return normalized


def export_train_eval_sample_final(path: Path) -> list[Path]:
    data = load_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path.name} should be a dict, got {type(data)!r}")

    exported_paths: list[Path] = []
    for split_name, split_data in data.items():
        if not isinstance(split_data, dict):
            raise TypeError(
                f"{path.name}[{split_name!r}] should be a dict, got {type(split_data)!r}"
            )

        output_path = OUTPUT_DIR / f"{path.stem}_{split_name}.csv"
        save_csv(pd.DataFrame(normalize_split_data(split_data)), output_path)
        exported_paths.append(output_path)
    return exported_paths


def export_feature_dict(path: Path) -> list[Path]:
    data = load_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path.name} should be a dict, got {type(data)!r}")

    rows = [{"feature": key, "value": value} for key, value in data.items()]
    output_path = OUTPUT_DIR / f"{path.stem}.csv"
    save_csv(pd.DataFrame(rows), output_path)
    return [output_path]


def export_vocab_dict(path: Path) -> list[Path]:
    data = load_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path.name} should be a dict, got {type(data)!r}")

    rows: list[dict[str, object]] = []
    for feature, vocab_values in data.items():
        for index, value in enumerate(vocab_values):
            rows.append({"feature": feature, "index": index, "value": value})

    output_path = OUTPUT_DIR / f"{path.stem}.csv"
    save_csv(pd.DataFrame(rows), output_path)
    return [output_path]


def export_pickle_to_csv(path: Path) -> list[Path]:
    if path.name == "train_eval_sample_final.pkl":
        return export_train_eval_sample_final(path)
    if path.name == "feature_dict.pkl":
        return export_feature_dict(path)
    if path.name == "vocab_dict.pkl":
        return export_vocab_dict(path)
    raise ValueError(f"Unsupported pickle file: {path.name}")


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed data directory not found: {DATA_DIR}")

    for file_name in PKL_FILES:
        pickle_path = DATA_DIR / file_name
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        exported_paths = export_pickle_to_csv(pickle_path)
        for output_path in exported_paths:
            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
