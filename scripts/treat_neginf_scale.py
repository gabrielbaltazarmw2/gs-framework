#!/usr/bin/env python3
"""T8: entrada tratada do FCGS para PLYs da CTC com escala -inf (decisão de 24/11, regra fixada antes do teste).

Regra: em cada eixo de escala (scale_*), cada -inf vira o mínimo finito daquele eixo na própria cena.
Nenhum outro atributo ou valor muda. NaN, +inf ou -inf fora de scale_* abortam (o caso volta ao chat central).
Se não houver -inf, nada é gravado: a entrada do FCGS é o PLY original (a regra é a identidade).

Uso (env fcgs, que tem numpy e plyfile):
  python scripts/treat_neginf_scale.py <cena> <frame>
Lê   ~/mpeg/datasets/<cena>/plys/<frame>.ply          (original da CTC, somente leitura; é a referência da avaliação)
Grava ~/mpeg/derived/<cena>/fcgs_input/<frame>.ply     (entrada tratada, só para o encoder; fora do git)
      ~/mpeg/derived/<cena>/fcgs_input/<frame>.json    (md5 do original e do tratado, trocas por eixo, commit)
"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
from plyfile import PlyData

H = Path.home() / "mpeg"
FW = H / "framework"

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main(cena, frame):
    src = H / "datasets" / cena / "plys" / f"{frame}.ply"
    out_dir = H / "derived" / cena / "fcgs_input"
    dst, meta = out_dir / f"{frame}.ply", out_dir / f"{frame}.json"
    ply = PlyData.read(str(src))
    v = ply["vertex"]
    for p in v.data.dtype.names:
        x = np.asarray(v[p], np.float64)
        ruim = np.isnan(x) | np.isposinf(x) | (np.isneginf(x) & (not p.startswith("scale_")))
        if ruim.any():
            sys.exit(f"ABORTADO: {int(ruim.sum())} valores não tratáveis em {p} (regra de 24/11: volta ao chat central)")
    trocas = {}
    for p in [n for n in v.data.dtype.names if n.startswith("scale_")]:
        x = v[p]
        m = np.isneginf(x)
        if m.any():
            mn = x[np.isfinite(x)].min()
            x[m] = mn
            trocas[p] = {"n": int(m.sum()), "valor": float(mn)}
    if not trocas:
        print(f"{cena} {frame}: nenhum -inf em escala; a entrada do FCGS é o original (nada gravado)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ply.write(str(dst))
    w = PlyData.read(str(dst))["vertex"]
    assert all(np.isfinite(np.asarray(w[p], np.float64)).all() for p in w.data.dtype.names), "saída com valor não finito"
    o = PlyData.read(str(src))["vertex"]
    mudou = [p for p in w.data.dtype.names if not p.startswith("scale_") and not np.array_equal(np.asarray(w[p]), np.asarray(o[p]))]
    assert not mudou, f"atributos alterados indevidamente: {mudou}"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=FW, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=FW, capture_output=True, text=True).stdout.strip())
    info = {"cena": cena, "frame": frame, "regra": "scale_* -inf -> mínimo finito do eixo na própria cena (decisão T8 de 24/11)",
            "original": {"path": str(src), "md5": md5(src)}, "tratado": {"path": str(dst), "md5": md5(dst)},
            "trocas": trocas, "framework_commit": commit, "framework_dirty": dirty}
    meta.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
