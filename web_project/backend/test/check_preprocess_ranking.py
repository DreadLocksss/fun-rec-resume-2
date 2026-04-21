from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path
import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from offline.config import config

    DATA_DIR = config.TEMP_DIR
except Exception:
    DATA_DIR = BACKEND_DIR.parent / "data" / "processed_data" / "web_project"

OUTPUT_DIR = DATA_DIR / "csv_exports"
PKL_FILES = (
    "ranking_train_eval_sample.pkl",
    "ranking_feature_dict.pkl",
    "ranking_vocab_dict.pkl",
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


def export_ranking_train_eval_sample(path: Path) -> list[Path]:
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
        save_csv(pd.DataFrame(split_data), output_path)
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
    if path.name == "ranking_train_eval_sample.pkl":
        return export_ranking_train_eval_sample(path)
    if path.name == "ranking_feature_dict.pkl":
        return export_feature_dict(path)
    if path.name == "ranking_vocab_dict.pkl":
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
