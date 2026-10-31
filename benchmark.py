"""Measure model loading, stage latency, end-to-end latency, and throughput."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from orchestrator import FiveAgentSystem
try:
    import psutil
except ImportError:
    psutil = None

def load_queries(path: str, limit: int):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        column = "instruction" if "instruction" in (rows.fieldnames or []) else "query"
        if column not in (rows.fieldnames or []): raise ValueError("Input requires an instruction or query column")
        return [row.get(column) or "" for row in rows][:limit]

def run(model: str, input_path: str, limit: int = 500):
    process = psutil.Process() if psutil else None
    rss_before = process.memory_info().rss if process else None
    started = time.perf_counter(); system = FiveAgentSystem.from_joblib(model); load = time.perf_counter() - started
    rss_after_load = process.memory_info().rss if process else None
    queries = load_queries(input_path, limit)
    if not queries: raise ValueError("No queries to benchmark")
    total_ms=[]; a1_ms=[]; a2_ms=[]
    for query in queries:
        t=time.perf_counter(); intent=system._analyze(query); a1_ms.append((time.perf_counter()-t)*1000)
        t=time.perf_counter(); system._retrieve(query, intent); a2_ms.append((time.perf_counter()-t)*1000)
        t=time.perf_counter(); system.process_query(query); total_ms.append((time.perf_counter()-t)*1000)
    elapsed=sum(total_ms)/1000
    return {"model": str(Path(model).resolve()), "queries": len(queries), "model_load_seconds": load,
            "agent1_latency_ms_median": statistics.median(a1_ms), "agent2_latency_ms_median": statistics.median(a2_ms),
            "end_to_end_latency_ms_median": statistics.median(total_ms), "end_to_end_latency_ms_p95": sorted(total_ms)[min(len(total_ms)-1, int(.95*len(total_ms)))],
            "throughput_queries_per_second": len(queries)/elapsed if elapsed else None,
            "process_rss_mb_before_model_load": round(rss_before/(1024*1024),2) if rss_before is not None else None,
            "process_rss_mb_after_model_load": round(rss_after_load/(1024*1024),2) if rss_after_load is not None else None,
            "model_load_rss_delta_mb": round((rss_after_load-rss_before)/(1024*1024),2) if rss_before is not None else None,
            "memory_measurement": "process RSS via psutil" if process else "unavailable (install psutil)"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model", default="agent_models/two_agent_system.joblib"); p.add_argument("--input", required=True); p.add_argument("--limit", type=int, default=500); p.add_argument("--output", default="outputs/benchmark.json")
    a=p.parse_args(); result=run(a.model,a.input,a.limit); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
