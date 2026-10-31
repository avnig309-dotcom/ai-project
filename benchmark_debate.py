"""Same-input ablation: legacy two-agent, five-agent/no debate, and debate."""
from __future__ import annotations
import argparse, csv, json, random, re, statistics, time
from pathlib import Path
from agents_pipeline import load_system, template_group
from orchestrator import FiveAgentSystem

UNSAFE = re.compile(r"\b(refund (?:has been|was) (?:processed|issued)|payment (?:was|has been) (?:made|refunded)|order (?:has been|was) cancelled|account (?:has been|was) (?:updated|deleted)|i (?:have|i've) (?:processed|issued|cancelled|updated|deleted))\b", re.I)

def run(model_path: str, cases_path: str, limit: int | None = None, seed: int = 42, train_path: str | None = None):
    baseline = load_system(model_path)
    no_debate = FiveAgentSystem(baseline, debate_enabled=False)
    debate = FiveAgentSystem(baseline, debate_enabled=True)
    records=[]
    with open(cases_path, encoding="utf-8-sig", newline="") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not ({"instruction"} <= set(reader.fieldnames)):
            raise ValueError("Cases need an instruction column")
        rows=list(reader)
    excluded_templates=0
    if train_path:
        with open(train_path, encoding="utf-8-sig", newline="") as f:
            train_templates={template_group(r.get("instruction") or "") for r in csv.DictReader(f)}
        filtered=[]
        for row in rows:
            if template_group(row.get("instruction") or "") in train_templates: excluded_templates+=1
            else: filtered.append(row)
        rows=filtered
    eligible_rows=len(rows)
    rng=random.Random(seed)
    if limit is not None and len(rows)>limit:
        groups={}
        for row in rows: groups.setdefault(row.get("intent") or "",[]).append(row)
        for bucket in groups.values(): rng.shuffle(bucket)
        sampled=[]
        while len(sampled)<limit:
            moved=False
            for key in sorted(groups):
                if groups[key] and len(sampled)<limit:
                    sampled.append(groups[key].pop()); moved=True
            if not moved: break
        rows=sampled
    for row in rows:
            query=row.get("instruction") or ""
            expected_intent=row.get("intent") or ""
            expected_decision=row.get("expected_decision") or ""
            samples=[]
            t=time.perf_counter(); old=baseline.ask(query); old_ms=(time.perf_counter()-t)*1000
            old_decision={"Clarification Required":"CLARIFY","Needs Information":"CLARIFY","No Confident Match":"ESCALATE"}.get(old.get("status"), "ANSWER")
            old_intent=(old.get("analysis") or {}).get("intent")
            samples.append(("A_two_agent",old_intent,old_decision,old.get("response", ""),old_ms,False))
            for name, system in (("B_five_agent_no_debate",no_debate),("C_five_agent_debate",debate)):
                t=time.perf_counter(); result=system.process_query(query); ms=(time.perf_counter()-t)*1000
                aud=result["agents"]["agent4"]
                unsafe=any(x.get("code")=="UNVERIFIED_TRANSACTION_CLAIM" for x in aud.get("issues",[]))
                samples.append((name,result.get("predicted_intent"),result["decision"],result["response"],ms,unsafe))
            records.append({"expected_intent":expected_intent,"expected_decision":expected_decision,"samples":samples})
    systems={}
    for name in ("A_two_agent","B_five_agent_no_debate","C_five_agent_debate"):
        relevant=[r for r in records]
        sample_list=[next(s for s in r["samples"] if s[0]==name) for r in relevant]
        intent_rows=[(r,s) for r,s in zip(relevant,sample_list) if r["expected_intent"]]
        decision_rows=[(r,s) for r,s in zip(relevant,sample_list) if r["expected_decision"]]
        systems[name]={"queries":len(sample_list),"intent_accuracy":(sum(s[1]==r["expected_intent"] for r,s in intent_rows)/len(intent_rows) if intent_rows else None),
            "decision_accuracy":(sum(s[2]==r["expected_decision"] for r,s in decision_rows)/len(decision_rows) if decision_rows else None),
            "clarification_precision":(sum(s[2]=="CLARIFY" and r["expected_decision"]=="CLARIFY" for r,s in decision_rows)/sum(s[2]=="CLARIFY" for r,s in decision_rows) if decision_rows and any(s[2]=="CLARIFY" for r,s in decision_rows) else None),
            "escalation_precision":(sum(s[2]=="ESCALATE" and r["expected_decision"]=="ESCALATE" for r,s in decision_rows)/sum(s[2]=="ESCALATE" for r,s in decision_rows) if decision_rows and any(s[2]=="ESCALATE" for r,s in decision_rows) else None),
            "expected_decision_labeled_rows":len(decision_rows),"unsupported_completed_action_claim_rate":sum(bool(UNSAFE.search(s[3])) for s in sample_list)/len(sample_list) if sample_list else None,
            "answer_rate":sum(s[2]=="ANSWER" for s in sample_list)/len(sample_list) if sample_list else None,
            "clarify_rate":sum(s[2]=="CLARIFY" for s in sample_list)/len(sample_list) if sample_list else None,
            "escalate_rate":sum(s[2]=="ESCALATE" for s in sample_list)/len(sample_list) if sample_list else None,
            "median_latency_ms":statistics.median(s[4] for s in sample_list) if sample_list else None,"api_calls_per_query":0}
    return {"sample_seed":seed,"eligible_rows_after_template_filter":eligible_rows,"template_overlap_rows_excluded":excluded_templates,"systems":systems,"limitations":["Decision quality is null unless cases contain manually reviewed expected_decision labels.","Intent accuracy uses only supplied intent labels.","Unsupported-claim rate detects explicit phrases and is not a semantic guarantee.","A uses legacy status-to-decision mapping for comparability."]}

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",default="agent_models/two_agent_system.joblib");p.add_argument("--cases",required=True);p.add_argument("--train");p.add_argument("--limit",type=int);p.add_argument("--seed",type=int,default=42);p.add_argument("--output",default="outputs/debate_ablation.json");a=p.parse_args()
    result=run(a.model,a.cases,a.limit,a.seed,a.train);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
