"""Run the five-agent system on unlabeled organizer queries without fitting."""
from __future__ import annotations
import argparse
import csv
import json
import itertools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from orchestrator import FiveAgentSystem

OUTPUT_COLUMNS = ["query_id", "predicted_intent", "predicted_category", "decision"]
DECISIONS = {"ANSWER", "CLARIFY", "ESCALATE"}
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{6,}\d)(?!\w)")
REFERENCE = re.compile(r"\b(?:order|account|payment|transaction|reference|ref)(?:\s*(?:id|number|no|ref|#))?\s*[:#]?\s*[A-Za-z0-9-]{5,}\b", re.I)

def redact_trace(value):
    if isinstance(value, dict): return {k: redact_trace(v) for k, v in value.items()}
    if isinstance(value, list): return [redact_trace(v) for v in value]
    if isinstance(value, str):
        value = EMAIL.sub("[REDACTED_EMAIL]", value)
        value = PHONE.sub("[REDACTED_PHONE]", value)
        return REFERENCE.sub("[REDACTED_REFERENCE]", value)
    return value

def validate_output(path: str | Path, expected_rows: int) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as check:
        reader = csv.DictReader(check)
        if reader.fieldnames != OUTPUT_COLUMNS: raise RuntimeError("Output CSV schema validation failed")
        rows = list(reader)
    if len(rows) != expected_rows: raise RuntimeError(f"Row count mismatch: expected={expected_rows}, output={len(rows)}")
    invalid = sorted({row.get("decision", "") for row in rows} - DECISIONS)
    if invalid: raise RuntimeError(f"Invalid decision value(s): {invalid}")
    return len(rows)

def run(input_path: str, output_path: str, model_path: str, trace_path: str | None = None, batch_size: int = 100) -> int:
    system = FiveAgentSystem.from_joblib(model_path)
    input_file = Path(input_path)
    output_file = Path(output_path); output_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file = Path(trace_path) if trace_path else output_file.with_suffix(".traces.jsonl")
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with input_file.open("r", encoding="utf-8-sig", newline="") as source, output_file.open("w", encoding="utf-8", newline="") as dest, trace_file.open("w", encoding="utf-8") as traces:
        reader = csv.DictReader(source)
        required = {"query_id", "instruction"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Input must contain columns {sorted(required)}; found {reader.fieldnames}")
        writer = csv.DictWriter(dest, fieldnames=OUTPUT_COLUMNS, lineterminator="\n"); writer.writeheader()
        size = max(1, batch_size)
        while True:
            chunk = list(itertools.islice(reader, size))
            if not chunk: break
            items = [{"query": row.get("instruction") or "", "query_id": (row.get("query_id") or "").strip()} for row in chunk]
            results = system.process_batch(items)
            for result in results:
                prediction = {"query_id": result["query_id"] or "", "predicted_intent": result["predicted_intent"] or "", "predicted_category": result["predicted_category"] or "", "decision": result["decision"]}
                if prediction["decision"] not in DECISIONS: raise ValueError(f"Invalid decision for row {total + 1}")
                writer.writerow(prediction)
                traces.write(json.dumps(redact_trace(result), ensure_ascii=False) + "\n")
                total += 1
            print(f"Processed {total} queries", file=sys.stderr)
    # Verify the exact required schema and one output per input data row.
    validate_output(output_file, total)
    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(ROOT / "agent_models" / "two_agent_system.joblib"))
    parser.add_argument("--traces"); parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    count = run(args.input, args.output, args.model, args.traces, args.batch_size)
    print(f"Wrote {count} predictions to {args.output}")

if __name__ == "__main__": main()
