#!/usr/bin/env python
"""T18 p1, validação na workstation: para cada cena e λ, codifica com --sync_shapes 0 e 1 (mesmo
--mem_sched e --chunk_div), decodifica cada bitstream e compara, arquivo por arquivo, o que o encoder
codificou com o que o decoder devolveu (t16_roundtrip.py + t16_compare.py). Também compara os bytes
de sync1 com os de sync0. Nada é gravado em results/.

O decoder da árvore do FCGS precisa ser o oficial (o script confere contra o 31e59a46).

Uso:
  python t18_validate.py --fcgs ~/mpeg/methods/fcgs --datasets ~/mpeg/datasets --scratch ~/mpeg/scratch/t18_p1 \
      --scenes manwithfruit_tracked:frame_0081 flowerdance:frame_0000 breakdance_untracked:frame_0000
"""
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--fcgs", type=Path, required=True)
ap.add_argument("--datasets", type=Path, required=True)
ap.add_argument("--scratch", type=Path, required=True)
ap.add_argument("--scenes", nargs="+", required=True, help="cena:frame")
ap.add_argument("--lmds", default="1e-4,2e-4,4e-4,8e-4,16e-4")
ap.add_argument("--chunk_div", type=int, default=4)
ap.add_argument("--mem_sched", type=int, default=2)
a = ap.parse_args()
fcgs, scr = a.fcgs.expanduser(), a.scratch.expanduser()
import torch  # noqa: E402

# o decoder da árvore tem de ser o oficial
cur = (fcgs / "model/FCGS_model.py").read_text()
off = subprocess.run(["git", "show", "31e59a46:model/FCGS_model.py"], cwd=fcgs, capture_output=True, text=True).stdout
assert cur[cur.index("def decomprss"):] == off[off.index("def decomprss"):], "o decoder da árvore difere do oficial"
assert "sync_shapes" in cur, "a árvore não tem o patch da T18"
print("decoder da árvore == oficial 31e59a46 ✓")

env = {**os.environ, "T16_PROBS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=fcgs, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
def bitfiles(d):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}
def recon(side):
    st = sorted(side.glob("recon_step*.pt"), key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))
    rs = [torch.load(p) for p in st]
    return torch.cat([r["xyz"] for r in rs]), torch.cat([r["fea"] for r in rs])

rows = []
for spec in a.scenes:
    scene, frame = spec.split(":")
    ply = a.datasets.expanduser() / scene / "plys" / f"{frame}.ply"
    assert ply.exists(), ply
    for lam in a.lmds.split(","):
        bits_by = {}
        for sync in (0, 1):
            w = scr / f"{scene}_{lam}_sync{sync}"
            shutil.rmtree(w, ignore_errors=True); (w / "bits").mkdir(parents=True)
            dump = w / "dump"
            rec = {"scene": scene, "lmd": lam, "sync": sync, "chunk_div": a.chunk_div, "mem_sched": a.mem_sched}
            if run([sys.executable, str(HERE / "t16_roundtrip.py"), "enc", str(dump), "encode_single_scene.py", "--lmd", lam,
                    "--determ", "1", "--chunk_div", str(a.chunk_div), "--mem_sched", str(a.mem_sched), "--sync_shapes", str(sync),
                    "--ply_path_from", str(ply), "--bit_path_to", str(w / "bits")], w / "enc.log"):
                rec["status"] = f"encode falhou ({w/'enc.log'})"; rows.append(rec); print(rec); continue
            if run([sys.executable, str(HERE / "t16_roundtrip.py"), "dec", str(dump), "decode_single_scene.py", "--lmd", lam,
                    "--chunk_div", str(a.chunk_div), "--bit_path_from", str(w / "bits"), "--ply_path_to", str(w / "dec.ply")], w / "dec.log"):
                rec["status"] = f"decode falhou ({w/'dec.log'})"; rows.append(rec); print(rec); continue
            subprocess.run([sys.executable, str(HERE / "t16_compare.py"), str(dump), "--json", str(w / "compare.json")],
                           stdout=open(w / "compare.txt", "w"), stderr=subprocess.STDOUT)
            cj = json.loads((w / "compare.json").read_text())
            ex, ef = recon(dump / "enc"); dx, df = recon(dump / "dec")
            rec.update(first_param=cj["first_param"], first_sym=cj["first_sym"],
                       recon_equal=bool(torch.equal(ex, dx) and torch.equal(ef, df)))
            rec["status"] = "EXATO" if rec["first_param"] is None and rec["first_sym"] is None and rec["recon_equal"] else "DIVERGE"
            bits_by[sync] = bitfiles(w / "bits")
            rec["bytes"] = sum(len(v) for v in bits_by[sync].values())
            shutil.rmtree(dump, ignore_errors=True)
            rows.append(rec)
            print(f"{scene:22s} {lam:6s} sync{sync}  {rec['status']:8s} 1º .b={rec['first_param']}  recon_igual={rec['recon_equal']}  {rec['bytes']} B", flush=True)
        if len(bits_by) == 2:
            same = bits_by[0] == bits_by[1]
            diff = sorted(k for k in set(bits_by[0]) | set(bits_by[1]) if bits_by[0].get(k) != bits_by[1].get(k))
            rows.append({"scene": scene, "lmd": lam, "sync_bytes_identical": same, "diff_files": diff[:12]})
            print(f"{'':22s} {lam:6s} bytes sync1 == sync0: {same}  {'' if same else diff[:6]}", flush=True)

scr.mkdir(parents=True, exist_ok=True)
(scr / "t18_validate.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
print(f"\nresumo em {scr / 't18_validate.json'}")
