"""
Trace verifier prototype for SRK outputs.
Compares adapter output to a canonical derivation (simple rules) and returns reward and trace_score.
"""
import re, json, importlib.util
spec = importlib.util.spec_from_file_location("sanskrit_tokenizer_v1", "/mnt/data/sanskrit_tokenizer_v1.py")
tkmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tkmod)

def canonical_derivation(input_iast):
    if hasattr(tkmod, "iast_tokenize_for_translit"):
        toks = tkmod.iast_tokenize_for_translit(input_iast)
    else:
        toks = [t.text for t in tkmod.iast_tokenize(input_iast)]
    toks = [t for t in toks if t and t!=" "]
    i = 0; trace = []
    while i < len(toks)-1:
        a = toks[i]; b = toks[i+1]; applied=False
        if isinstance(a,str) and a.endswith("a") and isinstance(b,str) and b.startswith("i"):
            new = a[:-1] + "e"; trace.append({"rule":"a+i->e","index":i,"before":[a,b],"after":new}); toks[i]=new; toks.pop(i+1); applied=True
        elif isinstance(a,str) and a.endswith("a") and isinstance(b,str) and b.startswith("u"):
            new = a[:-1] + "o"; trace.append({"rule":"a+u->o","index":i,"before":[a,b],"after":new}); toks[i]=new; toks.pop(i+1); applied=True
        elif a == "n" and isinstance(b,str) and b and b[0] in ("p","b"):
            trace.append({"rule":"n->m_before_pb","index":i,"before":[a,b],"after":["m",b]}); toks[i]="m"; applied=True
        else:
            i += 1
    final = "".join(toks)
    return final, trace

def verify(adapter_output):
    text = adapter_output.get("text","")
    # extract candidate final form (first token after colon or the last word)
    m = re.search(r'[:]\s*([^\s(]+)', text)
    claimed = m.group(1).strip() if m else None
    trace = adapter_output.get("srk_trace",{})
    input_tokens = trace.get("input_tokens", [])
    input_iast = " ".join(input_tokens)
    canonical_final, canonical_trace = canonical_derivation(input_iast)
    match = (claimed == canonical_final)
    reward = 1 if match else 0
    trace_score = 1.0 if match else 0.0
    return {"reward":reward, "trace_score":trace_score, "reason":{"claimed":claimed,"canonical":canonical_final,"match":match}, "canonical_trace":canonical_trace}

if __name__ == "__main__":
    sample = {"text":"Final form: be","srk_trace":{"input_tokens":["a","i"]}}
    print(verify(sample))