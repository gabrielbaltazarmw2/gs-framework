#!/usr/bin/env python
"""T16 p3 (2): verificação de validade dos pontos do FCGS, na máquina que gerou o ponto.

Para cada ponto em results/fcgs/<cena>/<frame>/lmd<λ>/ (só leitura; nada é gravado em results/):
  1. reencoda com os parâmetros do manifest (λ, chunk_div, mem_sched, entrada do encoder), gravando o
     que o encoder codifica (t16_roundtrip.py enc) -> o encode é determinístico, então o bitstream
     novo tem de ser idêntico ao guardado;
  2. decodifica o bitstream GUARDADO, gravando o que o decoder devolve (t16_roundtrip.py dec);
  3. compara símbolos e parâmetros arquivo por arquivo (t16_compare.py) -> primeiro .b divergente;
  4. compara a reconstrução do encoder com a saída do decoder e com o decoded.ply guardado, bit a bit.
Ponto EXATO = bitstream reproduzido, ida e volta exata e reconstrução do encoder == decoded.ply.

Uso:
  python t16_validate_points.py --fcgs ~/mpeg/methods/fcgs --results ~/mpeg/results/fcgs \
      --scratch ~/mpeg/scratch/t16_valid --points breakdance_untracked:all bartender_semitracked:1e-4,2e-4
Rodar com a GPU livre (reencode e decode usam a GPU). --keep_dumps guarda os dumps de cada ponto.
"""
import argparse, json, shutil, subprocess, sys, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--fcgs", type=Path, required=True)
ap.add_argument("--results", type=Path, required=True, help="results/fcgs")
ap.add_argument("--scratch", type=Path, required=True)
ap.add_argument("--points", nargs="+", required=True, help="cena:all ou cena:1e-4,2e-4")
ap.add_argument("--keep_dumps", action="store_true")
ap.add_argument("--sync_shapes", type=int, default=None, help="T18: força --sync_shapes no reencode (sem isto, usa o do manifest)")
a = ap.parse_args()
fcgs, res, scr = a.fcgs.expanduser(), a.results.expanduser(), a.scratch.expanduser()
LAMBDAS = ["1e-4", "2e-4", "4e-4", "8e-4", "16e-4"]
HEADER = "fcgs_header.bin"
CANON = (["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)]
         + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])

import torch  # noqa: E402
from plyfile import PlyData  # noqa: E402

def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=fcgs, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "T16_PROBS": "1"})
    return r.returncode

def files(d):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file() and p.name != HEADER}

def recon(dump_side):
    st = sorted(dump_side.glob("recon_step*.pt"), key=lambda p: int(p.stem[len("recon_step"):]))
    rs = [torch.load(p) for p in st]
    return torch.cat([r["xyz"] for r in rs]), torch.cat([r["fea"] for r in rs])

def to_canon(xyz, fea):
    n = xyz.shape[0]
    f_rest = fea[:, 3:48].reshape(n, 15, 3).transpose(1, 2).reshape(n, 45)   # como save_ply (transpose(1, 2))
    cols = [xyz, fea[:, 0:3], f_rest, fea[:, 48:49], fea[:, 49:52], fea[:, 52:56]]
    return torch.cat(cols, 1).numpy().astype(np.float32)

rows = []
for spec in a.points:
    scene, lams = spec.split(":")
    lams = LAMBDAS if lams == "all" else lams.split(",")
    for lam in lams:
        cand = sorted(res.glob(f"{scene}/*/lmd{lam}"))
        rec = {"scene": scene, "lmd": lam}
        if len(cand) != 1:
            rec["status"] = f"ponto não encontrado ({len(cand)} candidatos)"; rows.append(rec); print(rec); continue
        pdir = cand[0]
        man = json.loads((pdir / "manifest.json").read_text())
        cp = man.get("codec_params", {})
        div, ms = int(cp.get("chunk_div", 1)), int(cp.get("mem_sched", 0))
        ss = a.sync_shapes if a.sync_shapes is not None else int(cp.get("sync_shapes", 0))
        ei = man.get("encoder_input") or {}
        enc_in = (ei.get("path") if isinstance(ei, dict) else None) or man["input"]["path"]
        rec.update(frame=pdir.parent.name, chunk_div=div, mem_sched=ms, sync_shapes=ss, enc_input=enc_in, method_version=man.get("method_version"))
        work = scr / f"{scene}_{pdir.parent.name}_lmd{lam}"
        shutil.rmtree(work, ignore_errors=True); (work / "bits").mkdir(parents=True)
        dump = work / "dump"
        rc = run([sys.executable, str(HERE / "t16_roundtrip.py"), "enc", str(dump), "encode_single_scene.py",
                  "--lmd", lam, "--chunk_div", str(div), "--determ", "1", "--mem_sched", str(ms), "--sync_shapes", str(ss),
                  "--ply_path_from", enc_in, "--bit_path_to", str(work / "bits")], work / "enc.log")
        if rc:
            rec["status"] = f"reencode falhou (ver {work/'enc.log'})"; rows.append(rec); print(rec); continue
        old, new = files(pdir / "bitstream"), files(work / "bits")
        rec["bitstream_reproduzido"] = old == new
        rec["arquivos_diferentes"] = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))[:10]
        rc = run([sys.executable, str(HERE / "t16_roundtrip.py"), "dec", str(dump), "decode_single_scene.py",
                  "--lmd", lam, "--chunk_div", str(div), "--bit_path_from", str(pdir / "bitstream"),
                  "--ply_path_to", str(work / "dec.ply")], work / "dec.log")
        if rc:
            rec["status"] = f"decode falhou (ver {work/'dec.log'})"; rows.append(rec); print(rec); continue
        r = subprocess.run([sys.executable, str(HERE / "t16_compare.py"), str(dump), "--json", str(work / "compare.json")],
                           capture_output=True, text=True)
        (work / "compare.txt").write_text(r.stdout + r.stderr)
        cj = json.loads((work / "compare.json").read_text())
        rec["primeiro_b_param"], rec["primeiro_b_simbolo"] = cj["first_param"], cj["first_sym"]
        ex, ef = recon(dump / "enc"); dx, df = recon(dump / "dec")
        rec["recon_enc_igual_dec"] = bool(torch.equal(ex, dx) and torch.equal(ef, df))
        ply = PlyData.read(str(pdir / "decoded.ply"))["vertex"].data
        stored = np.stack([ply[c] for c in CANON], 1).astype(np.float32)
        mine = to_canon(ex, ef)
        rec["recon_enc_igual_decoded_ply"] = bool(stored.shape == mine.shape and np.array_equal(stored, mine))
        if stored.shape == mine.shape and not rec["recon_enc_igual_decoded_ply"]:
            rec["linhas_diferentes_no_ply"] = int((stored != mine).any(1).sum())
        exato = rec["bitstream_reproduzido"] and cj["first_param"] is None and cj["first_sym"] is None and rec["recon_enc_igual_decoded_ply"]
        rec["status"] = "EXATO" if exato else "DIVERGENTE"
        if not a.keep_dumps:
            shutil.rmtree(dump, ignore_errors=True)
        rows.append(rec)
        print(f"{scene:22s} {lam:6s} div{div} ms{ms} ss{ss}  {rec['status']:10s} bitstream reproduzido={rec['bitstream_reproduzido']}  "
              f"1º .b (param/símb)={rec['primeiro_b_param']}/{rec['primeiro_b_simbolo']}  enc==decoded.ply={rec['recon_enc_igual_decoded_ply']}", flush=True)

scr.mkdir(parents=True, exist_ok=True)
(scr / "t16_validate_points.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
print(f"\n{sum(r.get('status') == 'EXATO' for r in rows)} de {len(rows)} pontos EXATOS; detalhes em {scr / 't16_validate_points.json'}")
