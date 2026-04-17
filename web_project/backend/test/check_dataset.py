from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import pandas as pd


DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "funrec-movielens-1m"
OUTPUT_DIR = DATASET_DIR / "csv_exports"
PKL_FILES = (
    "movies.pkl",
    "ratings.pkl",
    "users.pkl",
    "movie_metadata.pkl",
)


def save_dataframe_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def export_pickle_file(pickle_path: Path, output_dir: Path) -> list[Path]:
    with pickle_path.open("rb") as file:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="numpy.core.numeric is deprecated",
                category=DeprecationWarning,
            )
            data = pickle.load(file)

    exported_paths: list[Path] = []

    if isinstance(data, pd.DataFrame):
        output_path = output_dir / f"{pickle_path.stem}.csv"
        save_dataframe_to_csv(data, output_path)
        exported_paths.append(output_path)
        return exported_paths

    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, pd.DataFrame):
                raise TypeError(
                    f"{pickle_path.name} contains an unsupported value for key "
                    f"{key!r}: {type(value)!r}"
                )

            output_path = output_dir / f"{pickle_path.stem}_{key}.csv"
            save_dataframe_to_csv(value, output_path)
            exported_paths.append(output_path)
        return exported_paths

    raise TypeError(f"Unsupported pickle content in {pickle_path.name}: {type(data)!r}")


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    for file_name in PKL_FILES:
        pickle_path = DATASET_DIR / file_name
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        exported_paths = export_pickle_file(pickle_path, OUTPUT_DIR)
        for output_path in exported_paths:
            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
