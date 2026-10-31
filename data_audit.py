"""Reproducible CSV schema, duplicate, template, and class-distribution audit."""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

def normalize(text: str) -> str:
    text = re.sub(r"\{\{.*?\}\}", " <slot> ", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

def normalize_literal(text: str) -> str:
    """Punctuation/whitespace normalization without replacing slot text."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())

def template(text: str) -> str:
    text = normalize(text)
    text = re.sub(r"\b\d+\b", " <num> ", text)
    return " ".join(text.split())

def read(path: str) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames: raise ValueError(f"Missing header: {path}")
        return list(reader), list(reader.fieldnames)

def audit(train_path: str, validation_path: str) -> dict:
    train, train_cols = read(train_path); val, val_cols = read(validation_path)
    train_norm = [normalize(x.get("instruction") or "") for x in train]
    val_norm = [normalize(x.get("instruction") or "") for x in val]
    train_literal = [normalize_literal(x.get("instruction") or "") for x in train]
    val_literal = [normalize_literal(x.get("instruction") or "") for x in val]
    train_templates = [template(x.get("instruction") or "") for x in train]
    val_templates = [template(x.get("instruction") or "") for x in val]
    train_set = set(train_norm); template_set = set(train_templates)
    train_literal_set = set(train_literal)
    train_labels = Counter(x.get("intent", "") for x in train)
    val_labels = Counter(x.get("intent", "") for x in val)
    conflicts = Counter()
    for row, norm in zip(train, train_norm): conflicts[norm] += 1
    normalized_label_sets: dict[str, set[str]] = {}
    for row, norm in zip(train, train_norm): normalized_label_sets.setdefault(norm, set()).add(row.get("intent", ""))
    return {
        "train": {"rows": len(train), "columns": train_cols, "intents": len(train_labels), "categories": len(set(x.get("category", "") for x in train)), "class_counts": dict(sorted(train_labels.items()))},
        "validation": {"rows": len(val), "columns": val_cols, "class_counts": dict(sorted(val_labels.items()))},
        "duplicates": {
            "train_repeated_normalized_rows": sum(n - 1 for n in conflicts.values() if n > 1),
            "train_repeated_template_rows": len(train_templates) - len(set(train_templates)),
            "validation_repeated_normalized_rows": len(val_norm) - len(set(val_norm)),
            "validation_normalized_overlap_with_train": sum(x in train_set for x in val_norm),
            "validation_literal_normalized_overlap_with_train": sum(x in train_literal_set for x in val_literal),
            "validation_template_overlap_with_train": sum(x in template_set for x in val_templates),
            "conflicting_labels_for_normalized_train_query_count": sum(len(v) > 1 for v in normalized_label_sets.values()),
        },
        "limitations": ["Template normalization is a reproducible heuristic, not semantic near-duplicate detection.", "Evaluation CSV is unlabeled and must not be used to select thresholds or estimate metrics."],
    }

def main():
    p = argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--validation", required=True); p.add_argument("--output", default="data_audit.json")
    a = p.parse_args(); result = audit(a.train, a.validation); Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
