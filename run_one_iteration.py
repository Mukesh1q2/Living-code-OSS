"""
Driver to run one prototype iteration of Challenger -> Solver -> Verifier.
"""
import json, random, time, importlib.util, os
BASE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("srk_adapter", os.path.join(BASE,"linguistic_core","srk_solver_adapter.py"))
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)
SRKClass = adapter_mod.SRKSolverAdapter

spec2 = importlib.util.spec_from_file_location("trace_verifier", os.path.join(BASE,"linguistic_core","trace_verifier.py"))
tv = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(tv)

with open(os.path.join(BASE,"rzero_custom","challenger_prompts_sanskrit.json"), "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

def sample_tasks(n=4):
    return random.sample(PROMPTS, min(n, len(PROMPTS)))

def run_iteration(num_samples_per_task=3):
    adapter = SRKClass()
    tasks = sample_tasks(4)
    results = []
    for t in tasks:
        print("="*60)
        print("CHALLENGER TASK TEMPLATE ID:", t.get("id"))
        prompt = t["template"]
        print(prompt)
        outputs = adapter.generate(prompt, num_samples=num_samples_per_task)
        for i,o in enumerate(outputs):
            print(f"--- Output #{i+1}: {o['text']}")
        verifications = [tv.verify(o) for o in outputs]
        avg_reward = sum(v["reward"] for v in verifications)/len(verifications)
        avg_trace_score = sum(v["trace_score"] for v in verifications)/len(verifications)
        print("AVERAGE REWARD:", avg_reward, "AVG TRACE SCORE:", avg_trace_score)
        results.append({"task":t,"outputs":outputs,"verifications":verifications})
    outdir = os.path.join(BASE,"runs"); os.makedirs(outdir, exist_ok=True)
    ts = int(time.time()); fname = os.path.join(outdir,f"run_{ts}.json")
    with open(fname,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print("Run summary saved to", fname)
    return results

if __name__ == "__main__":
    run_iteration()