"""Historical line-index splitter for auditing pre-cutover result artifacts."""

import os
from typing import Dict


def create_legacy_mod10_split(wordlist_path: str, output_dir: str) -> Dict[str, str]:
    """Reproduce the line-index split used by archived pre-cutover runs."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "train": os.path.join(output_dir, "data.train.wlh"),
        "validation": os.path.join(output_dir, "data.validation.wlh"),
        "test": os.path.join(output_dir, "data.test.wlh"),
    }
    counts = {key: 0 for key in paths}
    with (
        open(wordlist_path, encoding="utf-8") as wordlist,
        open(paths["train"], "w", encoding="utf-8") as train,
        open(paths["validation"], "w", encoding="utf-8") as validation,
        open(paths["test"], "w", encoding="utf-8") as test,
    ):
        for index, line in enumerate(wordlist):
            bucket = index % 10
            name = "train" if bucket < 8 else ("validation" if bucket == 8 else "test")
            if name == "train":
                train.write(line)
            elif name == "validation":
                validation.write(line)
            else:
                test.write(line)
            counts[name] += 1
    return {**paths, **{f"{key}_count": str(value) for key, value in counts.items()}}
