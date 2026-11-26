#!/usr/bin/env python3
"""T16 passo 1 — diagnóstico de integração do FCGS (só leitura; não altera nada).

Subcomandos:
  hash  <dirA> <dirB>            sha256 + bytes arquivo a arquivo de dois bitstreams
  file  <arq> [<arq> ...]        md5, sha256 e bytes de arquivos (ex.: PLY de entrada, decoded.ply)
  plydiff <plyA> <plyB>          diferença numérica por propriedade (usar se o sha256 do decoded divergir)
  stats <plyA> <plyB> [--names A B]  tabela de propriedades/faixas e checagens de convenção

Requer: numpy, plyfile (já presentes no env fcgs).
"""
import argparse, hashlib, os, sys
import numpy as np

SH_C0 = 0.28209479177387814


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(root):
    out = {}
    for d, _, fs in os.walk(root):
        for fn in fs:
            p = os.path.join(d, fn)
            out[os.path.relpath(p, root)] = (os.path.getsize(p), digest(p, "sha256"))
    return out


def cmd_hash(a):
    A, B = manifest(a.dirA), manifest(a.dirB)
    keys = sorted(set(A) | set(B))
    ok = True
    print(f"{'arquivo':40s} {'bytes A':>10s} {'bytes B':>10s}  sha256 igual?")
    for k in keys:
        sa, sb = A.get(k), B.get(k)
        same = sa is not None and sb is not None and sa[1] == sb[1]
        ok &= same
        print(f"{k:40s} {str(sa[0]) if sa else '—':>10s} {str(sb[0]) if sb else '—':>10s}  "
              f"{'sim' if same else 'NÃO'}"
              + ("" if same else f"   A={sa[1][:16] if sa else '-'} B={sb[1][:16] if sb else '-'}"))
    ta, tb = sum(v[0] for v in A.values()), sum(v[0] for v in B.values())
    print(f"\nTOTAL bytes: A={ta}  B={tb}  arquivos: A={len(A)} B={len(B)}")
    print("VEREDITO bitstream:", "IDÊNTICO" if ok and ta == tb else "DIFERENTE")
    return 0 if ok else 1


def cmd_file(a):
    for p in a.paths:
        print(f"{p}\n  bytes={os.path.getsize(p)}\n  md5={digest(p, 'md5')}\n  sha256={digest(p, 'sha256')}")


def load(path):
    from plyfile import PlyData
    v = PlyData.read(path).elements[0]
    names = [p.name for p in v.properties]
    return names, {n: np.asarray(v[n], dtype=np.float64) for n in names}


def cmd_plydiff(a):
    na, A = load(a.plyA)
    nb, B = load(a.plyB)
    print("mesma ordem de propriedades:", na == nb)
    if set(na) != set(nb):
        print("  só em A:", sorted(set(na) - set(nb)), " só em B:", sorted(set(nb) - set(na)))
    n = min(len(next(iter(A.values()))), len(next(iter(B.values()))))
    print(f"N: A={len(next(iter(A.values())))} B={len(next(iter(B.values())))}")
    worst = 0.0
    for k in na:
        if k in B:
            d = np.abs(A[k][:n] - B[k][:n])
            worst = max(worst, float(np.nanmax(d)))
            if d.max() > 0:
                print(f"  {k:12s} max|Δ|={d.max():.3e}  frac≠0={np.mean(d > 0):.4f}")
    print("VEREDITO decoded (numérico):", "IDÊNTICO" if worst == 0 else f"DIFERENTE (max|Δ|={worst:.3e})")


def group(names, D, prefix):
    ks = sorted([k for k in names if k.startswith(prefix)], key=lambda x: int(x.split("_")[-1]))
    return ks, (np.stack([D[k] for k in ks], 1) if ks else None)


def row(x):
    x = x[np.isfinite(x)]
    return f"[{x.min():9.4f}, {x.max():9.4f}]  μ={x.mean():8.4f}  σ={x.std():7.4f}"


def summarize(path):
    names, D = load(path)
    N = len(D[names[0]])
    r = {"names": names, "N": N}
    nonfinite = sum(int((~np.isfinite(D[k])).sum()) for k in names)
    r["nonfinite"] = nonfinite
    _, dc = group(names, D, "f_dc_")
    rk, rest = group(names, D, "f_rest_")
    _, sc = group(names, D, "scale_")
    _, ro = group(names, D, "rot_")
    op = D.get("opacity")
    xyz = np.stack([D["x"], D["y"], D["z"]], 1)
    r["n_rest"] = len(rk)
    r["rows"] = {
        "f_dc (todos)": row(dc.ravel()),
        **{f"f_dc_{i}": row(dc[:, i]) for i in range(dc.shape[1])},
        "f_rest (todos)": row(rest.ravel()) if rest is not None else "ausente",
        "opacity": row(op) if op is not None else "ausente",
        "scale (todos)": row(sc.ravel()),
        "|rot| (norma)": row(np.linalg.norm(ro, axis=1)),
    }
    chk = {}
    c = 0.5 + SH_C0 * dc
    chk["cor implícita 0.5+C0·f_dc fora de [0,1]"] = f"{np.mean((c < 0) | (c > 1)):.4f}"
    chk["f_dc inteiro em [0,1] (sugere cor RGB, não SH0)"] = str(bool(dc.min() >= 0 and dc.max() <= 1))
    chk["opacity inteira em [0,1] (sugere sigmoide já aplicada)"] = str(bool(op is not None and op.min() >= 0 and op.max() <= 1))
    chk["scale toda > 0 (sugere escala linear, não log)"] = str(bool(sc.min() > 0))
    if rest is not None and rest.shape[1] == 45:
        s = rest.std(0)
        # Heurística: em cada layout, o σ do mesmo coeficiente nos 3 canais deve ser parecido.
        # canal-major (3DGS/FCGS): R0..14, G0..14, B0..14 -> compara blocos de 15
        # intercalado: c0R c0G c0B c1R ... -> compara elementos de cada trinca
        m = s.mean()
        d_cm = (np.abs(s[0:15] - s[15:30]).mean() + np.abs(s[0:15] - s[30:45]).mean()) / (2 * m)
        d_il = (np.abs(s[0::3] - s[1::3]).mean() + np.abs(s[0::3] - s[2::3]).mean()) / (2 * m)
        chk["dist. σ f_rest se canal-major (menor = layout provável)"] = f"{d_cm:.3f}"
        chk["dist. σ f_rest se intercalado"] = f"{d_il:.3f}"
        r["rest_std"] = s
    mu, sd = xyz.mean(0), xyz.std(0, ddof=1)
    out = np.any(np.abs(xyz - mu) > 3 * sd, axis=1)
    chk["xyz fora de μ±3σ (mask do normalize_xyz, nr=3)"] = f"{out.mean():.4f}"
    chk["extensão xyz (σ por eixo)"] = np.array2string(sd, precision=3)
    r["chk"] = chk
    return r


def cmd_stats(a):
    nA, nB = a.names
    RA, RB = summarize(a.plyA), summarize(a.plyB)
    print(f"| item | {nA} | {nB} |\n|---|---|---|")
    print(f"| N gaussianas | {RA['N']} | {RB['N']} |")
    print(f"| nº propriedades / nº f_rest | {len(RA['names'])} / {RA['n_rest']} | {len(RB['names'])} / {RB['n_rest']} |")
    print(f"| mesma lista e ordem de propriedades | {RA['names'] == RB['names']} | |")
    print(f"| valores não finitos | {RA['nonfinite']} | {RB['nonfinite']} |")
    for k in RA["rows"]:
        print(f"| {k} | {RA['rows'][k]} | {RB['rows'].get(k, '—')} |")
    for k in RA["chk"]:
        print(f"| {k} | {RA['chk'][k]} | {RB['chk'].get(k, '—')} |")
    if RA["names"] != RB["names"]:
        print(f"\nordem {nA}: {RA['names']}\nordem {nB}: {RB['names']}")
    for n, R in ((nA, RA), (nB, RB)):
        if "rest_std" in R:
            print(f"\nσ f_rest_0..44 {n}: " + " ".join(f"{v:.3f}" for v in R["rest_std"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    h = s.add_parser("hash"); h.add_argument("dirA"); h.add_argument("dirB")
    f = s.add_parser("file"); f.add_argument("paths", nargs="+")
    d = s.add_parser("plydiff"); d.add_argument("plyA"); d.add_argument("plyB")
    t = s.add_parser("stats"); t.add_argument("plyA"); t.add_argument("plyB")
    t.add_argument("--names", nargs=2, default=["A", "B"])
    a = p.parse_args()
    sys.exit({"hash": cmd_hash, "file": cmd_file, "plydiff": cmd_plydiff, "stats": cmd_stats}[a.cmd](a) or 0)
