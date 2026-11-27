#!/usr/bin/env python
"""T16 p3 (diagnóstico, não altera o FCGS): roda encode_single_scene.py ou decode_single_scene.py
oficiais e grava, para cada arquivo .b, o que entrou no codificador (encoder) e o que saiu do
decodificador (decoder): símbolos e, opcionalmente, os parâmetros de probabilidade (média, escala, peso).
Também grava as posições que entram em normalize_xyz (as usadas nos contextos).

As funções de codificação são envolvidas no namespace de model.FCGS_model; as contas não mudam.

Uso (na raiz da árvore do FCGS):
  python t16_roundtrip.py enc <dump_dir> encode_single_scene.py --lmd 1e-4 ...   # argumentos do script oficial
  python t16_roundtrip.py dec <dump_dir> decode_single_scene.py --lmd 1e-4 ...
Variável T16_PROBS=0 desliga a gravação dos parâmetros (só símbolos).
"""
import os, runpy, sys
from pathlib import Path

side, dump = sys.argv[1], Path(sys.argv[2])
assert side in ("enc", "dec")
script, args = sys.argv[3], sys.argv[4:]
PROBS = os.environ.get("T16_PROBS", "1") != "0"
out = dump / side
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
import model.FCGS_model as M  # noqa: E402  (o script oficial importa o mesmo módulo)

def key(file_name):
    p = Path(file_name)
    return f"{p.parent.name}_{p.name}"          # "<step>_<arquivo>.b"

def cpu(t):
    return t.detach().float().cpu() if torch.is_tensor(t) else torch.tensor(float(t))

def params(k):
    rec = {}
    if PROBS:
        for n in ("mean_list", "scale_list", "prob_list"):
            if n in k:
                rec[n[:-5]] = torch.stack([cpu(t).view(-1) for t in k[n]])
        if "Q" in k:
            rec["Q"] = cpu(k["Q"]).view(-1)
    return rec

def wrap(fn, is_enc):
    def w(*a, **k):
        name = key(k["file_name"])
        if is_enc:
            rec = {"sym": cpu(k["x"]).view(-1), **params(k)}
            torch.save(rec, out / f"{name}.pt")
            return fn(*a, **k)
        res = fn(*a, **k)
        rec = {"sym": cpu(res).view(-1), **params(k)}
        torch.save(rec, out / f"{name}.pt")
        return res
    return w

for n in ("encoder", "encoder_factorized_chunk", "encoder_gaussian_mixed_chunk"):
    setattr(M, n, wrap(getattr(M, n), True))
for n in ("decoder", "decoder_factorized_chunk", "decoder_gaussian_mixed_chunk"):
    setattr(M, n, wrap(getattr(M, n), False))

_norm = M.normalize_xyz
_calls = {"n": 0}
def norm_wrap(xyz_orig, *a, **k):
    r = _norm(xyz_orig, *a, **k)
    torch.save({"xyz": cpu(xyz_orig), "norm_xyz_clamp": cpu(r[1])}, out / f"positions_call{_calls['n']}.pt")
    _calls["n"] += 1
    return r
M.normalize_xyz = norm_wrap

# reconstrução: o que o encoder devolve (compress) e o que o decoder devolve (decomprss), por passo
_rec = {"n": 0}
_compress, _decomprss = M.FCGS.compress, M.FCGS.decomprss
def compress_wrap(self, *a, **k):
    r = _compress(self, *a, **k)
    torch.save({"xyz": cpu(r[0]), "fea": cpu(r[1])}, out / f"recon_step{_rec['n']}.pt"); _rec["n"] += 1
    return r
def decomprss_wrap(self, *a, **k):
    r = _decomprss(self, *a, **k)
    torch.save({"xyz": cpu(r[0]), "fea": cpu(r[1])}, out / f"recon_step{_rec['n']}.pt"); _rec["n"] += 1
    return r
M.FCGS.compress, M.FCGS.decomprss = compress_wrap, decomprss_wrap

sys.argv = [script] + args
runpy.run_path(script, run_name="__main__")
print(f"[T16] {side}: {len(list(out.glob('*.pt')))} arquivos gravados em {out}")
