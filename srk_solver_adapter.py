"""
SRK Solver Adapter (prototype)

Exposes a generate(prompt, num_samples=1, temp=1.0) function that R-Zero can call.
Returns list of dicts: {"text": <English summary or code>, "srk_trace": <structured trace>}

This prototype uses a tiny deterministic rule engine for sandhi/assimilation.
It expects the tokenizer module at /mnt/data/sanskrit_tokenizer_v1.py (created earlier).
"""
import random, re, time
import importlib.util
spec = importlib.util.spec_from_file_location("sanskrit_tokenizer_v1", "/mnt/data/sanskrit_tokenizer_v1.py")
tkmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tkmod)

def apply_sandhi_simple(iast_tokens):
    toks = iast_tokens[:]
    trace = []
    i = 0
    while i < len(toks)-1:
        a = toks[i]; b = toks[i+1]; applied = False
        if isinstance(a,str) and a.endswith("a") and isinstance(b,str) and b.startswith("i"):
            new = a[:-1] + "e"
            trace.append({"rule":"a+i->e","index":i,"before":[a,b],"after":new})
            toks[i] = new
            toks.pop(i+1)
            applied = True
        elif isinstance(a,str) and a.endswith("a") and isinstance(b,str) and b.startswith("u"):
            new = a[:-1] + "o"
            trace.append({"rule":"a+u->o","index":i,"before":[a,b],"after":new})
            toks[i] = new
            toks.pop(i+1)
            applied = True
        elif a == "n" and isinstance(b,str) and b and b[0] in ("p","b"):
            trace.append({"rule":"n->m_before_pb","index":i,"before":[a,b],"after":["m",b]})
            toks[i] = "m"
            applied = True
        elif a == "ḥ" and isinstance(b,str) and b and b[0] in list("aiāīuūeorṛṝḷḹ"):
            trace.append({"rule":"visarga->s_before_vowel","index":i,"before":[a,b],"after":["s",b]})
            toks[i] = "s"
            applied = True
        if not applied:
            i += 1
    return toks, trace

class SRKSolverAdapter:
    def __init__(self):
        self.tokenizer = tkmod

    def _extract_input(self, prompt):
        m = re.search(r'INPUT:\s*"(.*?)"', prompt, re.S)
        if m:
            return m.group(1).strip()
        # fallback: if single token present
        lines = prompt.splitlines()
        for L in lines:
            if L.strip():
                return L.strip()
        return prompt.strip()

    def _attempt_derivation(self, input_iast):
        # tokenization: try to use iast_tokenize_for_translit if available, else iast_tokenize
        if hasattr(self.tokenizer, "iast_tokenize_for_translit"):
            toks = self.tokenizer.iast_tokenize_for_translit(input_iast)
        else:
            toks = [t.text for t in self.tokenizer.iast_tokenize(input_iast)]
        toks = [t for t in toks if t not in (" ","")]
        new_toks, trace = apply_sandhi_simple(toks)
        final = "".join(new_toks)
        summary = f"Final form: {final}"
        srk_trace = {"input_tokens": toks, "applied": trace, "final_tokens": new_toks}
        return summary, srk_trace

    def generate(self, prompt, num_samples=1, temp=1.0):
        input_str = self._extract_input(prompt)
        outputs = []
        for _ in range(num_samples):
            summary, trace = self._attempt_derivation(input_str)
            # small stochasticity in summary wording
            if random.random() < 0.2:
                summary = summary.replace("Final form:", "Result:")
            outputs.append({"text": summary, "srk_trace": trace, "meta":{"t": time.time()}})
        return outputs

# quick local test
if __name__ == "__main__":
    a = SRKSolverAdapter()
    print(a.generate('INPUT: "bhūmi + pāla"', num_samples=2))