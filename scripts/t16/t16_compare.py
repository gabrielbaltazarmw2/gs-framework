#!/usr/bin/env python
"""T16 p3: compara o que o encoder codificou com o que o decoder devolveu, arquivo por arquivo, na
ordem de codificação. Mostra onde aparece a primeira diferença nos parâmetros de probabilidade e nos
símbolos. Se os parâmetros divergem antes dos símbolos, a causa é a probabilidade diferente entre os
lados (dessincronização). Se os símbolos divergem com parâmetros idênticos, o problema está no
codificador aritmético.

Uso: python t16_compare.py <dump_dir> [--json saida.json]
"""
import json, sys
from pathlib import Path
import torch

dump = Path(sys.argv[1])
js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
E, D = dump / "enc", dump / "dec"

ORDER = (["mask.b", "g_fea_enc_q_hyp_q.b"]
         + [f"g_fea_enc_q_sp{s}_ch{c}.b" for s in range(4) for c in range(4)]
         + ["fe_q_hyp_q.b"] + [f"fe_q_sp{s}_ch{c}.b" for s in range(4) for c in range(3)]
         + ["geo_q_hyp_q.b"] + [f"geo_q_sp{s}.b" for s in range(4)])

def first_diff(a, b):
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    ne = (a != b) & ~(torch.isnan(a) & torch.isnan(b))
    n = int(ne.sum())
    if n == 0:
        return None
    idx = int(ne.any(0).nonzero()[0, 0]) if a.dim() > 1 else int(ne.nonzero()[0, 0])   # índice do elemento
    return {"n_diff": n, "frac": n / ne.numel(), "first_idx": idx,
            "max_abs": float((a - b).abs()[ne].max())}

rows, first_param, first_sym = [], None, None
steps = sorted({p.name.split("_", 1)[0] for p in E.glob("*_*.b.pt")})
print(f"passos: {steps}")

# posições
for pe in sorted(E.glob("positions_call*.pt")):
    pd = D / pe.name
    if pd.exists():
        a, b = torch.load(pe), torch.load(pd)
        for k in ("xyz", "norm_xyz_clamp"):
            d = first_diff(a[k], b[k])
            rows.append({"file": f"{pe.stem}:{k}", "pos": True, "diff": d})
            print(f"{pe.stem}:{k:15s} {'idênticas' if d is None else f'DIFEREM em {d}'}")

print(f"\n{'arquivo':28s} {'símbolos':40s} parâmetros (média | escala | peso)")
for st in steps:
    for name in ORDER:
        f = f"{st}_{name}.pt"
        if not (E / f).exists() or not (D / f).exists():
            continue
        a, b = torch.load(E / f), torch.load(D / f)
        ds = first_diff(a["sym"], b["sym"])
        dp = {k: first_diff(a[k], b[k]) for k in ("mean", "scale", "prob") if k in a and k in b}
        rows.append({"file": f"{st}_{name}", "sym": ds, "params": dp})
        if first_sym is None and ds is not None:
            first_sym = f"{st}_{name}"
        if first_param is None and any(v is not None for v in dp.values()):
            first_param = f"{st}_{name}"
        s_sym = "iguais" if ds is None else f"{ds.get('n_diff')} dif ({ds.get('frac', 0):.1%}), 1º idx {ds.get('first_idx')}"
        s_par = " | ".join("=" if dp.get(k) is None else f"≠ 1º {dp[k].get('first_idx')} máx {dp[k].get('max_abs', 0):.2e}"
                           for k in ("mean", "scale", "prob") if k in dp) or "(não gravados)"
        print(f"{st+'_'+name:28s} {s_sym:40s} {s_par}")

print("\nPRIMEIRA divergência nos parâmetros (ordem de codificação):", first_param)
print("PRIMEIRA divergência nos símbolos   (ordem de codificação):", first_sym)
if first_param is None and first_sym is None:
    print("VEREDITO: ida e volta exata; o decoder devolve exatamente o que o encoder codificou.")
elif first_sym is not None and first_param is not None and ORDER.index(first_param.split("_", 1)[1]) <= ORDER.index(first_sym.split("_", 1)[1]):
    print("VEREDITO: os parâmetros de probabilidade divergem antes dos símbolos -> dessincronização entre encoder e decoder.")
elif first_sym is not None:
    print("VEREDITO: símbolos divergem com parâmetros idênticos até ali -> olhar o codificador aritmético.")
else:
    print("VEREDITO: parâmetros diferem, mas os símbolos saíram iguais (diferença sem efeito nesta execução).")
if js:
    json.dump({"rows": rows, "first_param": first_param, "first_sym": first_sym}, open(js, "w"), indent=1)
