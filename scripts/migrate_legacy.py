#!/usr/bin/env python3
"""T2: migra resultados existentes (GSCS rp1-rp4, FCGS) para o layout do contrato.
Nao apaga nem altera nada na origem. Usa hardlink para arquivos grandes."""
import json, os, re, shutil
from pathlib import Path

MPEG = Path.home() / "mpeg"
OLD = MPEG / "gscodec_studio/examples/results"
NEW = MPEG / "results"
SCENE, FRAME = "breakdance_untracked", "frame_0000"

def put(src, dst):
    if not src.exists():
        print(f"  AVISO: nao encontrado {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def manifest(d, m):
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))

# GSCS (HM): bitstream = .mp4 + metadata json
for rp in ["rp1", "rp2", "rp3", "rp4"]:
    src = OLD / "ctc_1f/breakdance" / rp
    dst = NEW / "gscs_hm" / SCENE / FRAME / rp
    files = sorted((src / "compression").glob("*.mp4")) + \
            sorted((src / "compression").glob("*metadata*.json"))
    for f in files:
        put(f, dst / "bitstream" / f.name)
    put(src / "decoded_ply" / f"{FRAME}.ply", dst / "decoded.ply")
    put(src / "ctc_metrics.txt", dst / "legacy" / "ctc_metrics.txt")
    manifest(dst, {
        "method": "gscs_hm", "method_version": "gscodec_studio 12a0bcce",
        "scene": SCENE, "frame": FRAME,
        "rate_param": {"type": "rate_point", "value": rp},
        "conda_env": "gscodec_studio",
        "rate_source": "measured_bitstream",
        "n_bitstream_files": len(files),
    })
    print(f"GSCS {rp}: {len(files)} arquivos de bitstream")

# FCGS: sem bitstream, taxa do resumo (reported_summary)
for d in sorted((OLD / "FCGS/breakdance").glob("lambda_*")):
    summ = next(d.glob("bitstream_summary_*.txt"), None)
    if summ is None:
        print(f"FCGS {d.name}: AVISO sem resumo de bitstream")
        continue
    lam = re.search(r"lambda_(.+)\.txt", summ.name).group(1)
    entries = re.findall(r"([\d.]+)\s*KiB\s*\|\s*(\S+)", summ.read_text())
    rep_bytes = round(sum(float(k) for k, _ in entries) * 1024)
    dst = NEW / "fcgs" / SCENE / FRAME / d.name
    put(d / "decoded_ply" / f"{FRAME}.ply", dst / "decoded.ply")
    put(summ, dst / "bitstream_report" / summ.name)
    put(d / "ctc_metrics.txt", dst / "legacy" / "ctc_metrics.txt")
    manifest(dst, {
        "method": "fcgs", "method_version": None,
        "scene": SCENE, "frame": FRAME,
        "rate_param": {"type": "lambda", "value": lam},
        "conda_env": "google_colab",
        "rate_source": "reported_summary",
        "reported_rate_bytes": rep_bytes,
        "n_bitstream_files": len(entries),
        "notas": "bitstream nao disponivel localmente; taxa reconstruida do resumo (precisao 0.01 KiB por arquivo)",
    })
    print(f"FCGS {d.name}: lambda={lam}, {len(entries)} arquivos, {rep_bytes} bytes ({rep_bytes/2**20:.2f} MiB)")
