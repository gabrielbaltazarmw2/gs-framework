#!/usr/bin/env python3
"""T2: avaliacao CTC padronizada (Passos A, C, D, E) para qualquer metodo.

Uso:
  evaluate.py <dir_ponto> [<dir_ponto> ...]  avalia pontos especificos
  evaluate.py --all                           avalia todos os pontos em results/
  evaluate.py --summary                       tabela a partir dos metrics.json
Opcoes:
  --force         refaz a avaliacao mesmo se eval/metrics.json ja existir
  --check-legacy  compara o PSNR novo com legacy/ctc_metrics.txt
"""
import os
import argparse, json, re, subprocess, time
from datetime import datetime
from pathlib import Path

FW = Path(__file__).resolve().parent.parent
def load(p): return json.loads(Path(p).read_text())
# T12: GSC_PATHS escolhe outro paths.json (outra versao das ferramentas); sem ela, o padrao
PATHS_FILE = Path(os.environ.get("GSC_PATHS", FW / "configs/paths.json")).expanduser()
if not PATHS_FILE.is_absolute():
    PATHS_FILE = FW / PATHS_FILE
P = {k: (Path(v).expanduser() if isinstance(v, str) and v.startswith("~") else v)
     for k, v in load(PATHS_FILE).items()}
SCENES = load(FW / "configs/scenes.json")

SCALARS = {
    "psnr_rgb_avg":    r"^Psnr RGB \(avg\)\s*=\s*([\d.]+)",
    "psnr_yuv_avg":    r"^Psnr YUV \(avg\)\s*=\s*([\d.]+)",
    "om_psnr_rgb_avg": r"^OM-Psnr RGB \(avg\)\s*=\s*([\d.]+)",
    "om_psnr_yuv_avg": r"^OM-Psnr YUV \(avg\)\s*=\s*([\d.]+)",
    "ssim_avg":        r"^SSIM \(avg\)\s*=\s*([\d.]+)",
    "ivssim_rgb":      r"^IVSSIM RGB\s*=\s*([\d.]+)",
    "ivssim_yuv":      r"^IVSSIM YUV\s*=\s*([\d.]+)",
    "occupancy_pct":   r"^Occupancy\s*=\s*([\d.]+)%",
}
TRIPLES = {
    "psnr_rgb": r"^Psnr RGB\s*=\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)",
    "psnr_yuv": r"^Psnr YUV\s*=\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)",
}

def parse_metrics(text):
    """Usa a ultima ocorrencia de cada chave (bloco 'Metric Average')."""
    out = {}
    for k, rx in SCALARS.items():
        m = re.findall(rx, text, re.M)
        out[k] = float(m[-1]) if m else None
    for k, rx in TRIPLES.items():
        m = re.findall(rx, text, re.M)
        out[k] = [float(x) for x in m[-1]] if m else None
    return out

def run(cmd, log):
    cmd = [str(c) for c in cmd]
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"falhou (codigo {r.returncode}), ver {log}")

def camera(ply_in, ply_out, scene, log):
    sparse = P["datasets"] / scene / SCENES[scene]["sparse_dir"]
    sub = [P["camera_subcommand"]] if P.get("camera_subcommand") else []  # T12: "gstools camera" na 155
    run([P["camera_bin"], *sub, "-i", ply_in,
         f"--camera={sparse / 'cameras.bin'}", f"--image={sparse / 'images.bin'}",
         "-o", ply_out], log)

def ensure_source(scene, frame):
    """Passo A: source com camera, gerado uma vez por cena em derived/."""
    src = P["derived"] / scene / "plys_cam_info" / f"{frame}.ply"
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Passo A: gerando {src}")
        camera(P["datasets"] / scene / "plys" / f"{frame}.ply", src, scene,
               src.parent / f"{frame}_camera.log")
    return src

def evaluate(point, force=False, check=False):
    m = load(point / "manifest.json")
    scene, frame = m["scene"], m["frame"]
    sc = SCENES[scene]
    out = point / "eval"
    out.mkdir(exist_ok=True)
    mj = out / "metrics.json"
    label = f"{m['method']} {point.name}"

    if mj.exists() and not force:
        print(f"[{label}] ja avaliado, pulando (use --force para refazer)")
        res = load(mj)
    else:
        print(f"[{label}] avaliando")
        src = ensure_source(scene, frame)
        pos = out / "decoded_pos.ply"
        camera(point / "decoded.ply", pos, scene, out / "camera.log")          # Passo C
        txt = out / "ctc_metrics.txt"
        thr = min(sc['vistas'], os.cpu_count())   # N420/N1646: de 1 ao n. de vistas avaliadas
        t0 = time.time()
        run([P["metrics_bin"], "-a", src, "-b", pos,                            # Passo D
             "--frameCount=1", f"--startFrame={sc.get('start_frame', 0)}",
             "--useCameraPosition=1", "--cpu=1",
             f"--width={sc['width']}", f"--height={sc['height']}",
             f"--threads={thr}",           # N420: 1 a n. de vistas
             "-o", txt], out / "metrics.log")
        dur = time.time() - t0
        bs = point / "bitstream"                                                # Passo E
        files = [f for f in bs.rglob("*") if f.is_file()] if bs.is_dir() else []
        if files:
            rate, source, nfiles = sum(f.stat().st_size for f in files), "measured_bitstream", len(files)
        else:
            rate, source, nfiles = m.get("reported_rate_bytes"), m.get("rate_source", "unknown"), m.get("n_bitstream_files")
        res = {
            "method": m["method"], "scene": scene, "frame": frame,
            "rate_param": m["rate_param"],
            "rate_bytes": rate,
            "rate_MB": rate / 1e6 if rate else None,
            "rate_MiB": rate / 2**20 if rate else None,
            "rate_source": source, "n_rate_files": nfiles,
            "metrics": parse_metrics(txt.read_text()),
            "metrics_seconds": round(dur, 1),
            "metric_threads": thr,
            "tools_version": P["tools_version"],
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        mj.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"[{label}] taxa {res['rate_MB']:.3f} MB ({source}), "
              f"RGB {res['metrics']['psnr_rgb_avg']:.4f}, YUV {res['metrics']['psnr_yuv_avg']:.4f}, "
              f"{dur:.0f} s")

    leg = point / "legacy" / "ctc_metrics.txt"
    if check and leg.exists():
        lp = parse_metrics(leg.read_text())
        d_rgb = res["metrics"]["psnr_rgb_avg"] - lp["psnr_rgb_avg"]
        d_yuv = res["metrics"]["psnr_yuv_avg"] - lp["psnr_yuv_avg"]
        status = "IDENTICO" if d_rgb == 0 and d_yuv == 0 else "DIFERENTE"
        print(f"[{label}] legado: dRGB={d_rgb:+.3e} dYUV={d_yuv:+.3e} -> {status}")

def all_points():
    return sorted(p.parent for p in P["results"].glob("*/*/*/*/manifest.json"))

def summary():
    rows = []
    for pt in all_points():
        mj = pt / "eval" / "metrics.json"
        if mj.exists():
            r = load(mj)
            rows.append((r["method"], r["scene"], pt.name, r["rate_MB"], r["rate_source"],
                         r["metrics"]["psnr_rgb_avg"], r["metrics"]["psnr_yuv_avg"]))
        else:
            rows.append((pt.parts[-4], pt.parts[-3], pt.name, None, "nao avaliado", None, None))
    rows.sort(key=lambda r: (r[0], r[1], r[3] or 0))
    print(f"{'metodo':<10}{'cena':<24}{'ponto':<11}{'taxa MB':>9}  {'fonte':<20}{'RGB-PSNR':>9}{'YUV-PSNR':>10}")
    for r in rows:
        f = lambda x, n: f"{x:.{n}f}" if x is not None else "-"
        print(f"{r[0]:<10}{r[1]:<24}{r[2]:<11}{f(r[3],3):>9}  {r[4]:<20}{f(r[5],2):>9}{f(r[6],2):>10}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("points", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-legacy", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        if a.all and "GSC_PATHS" in os.environ:
            ap.error("--all com GSC_PATHS avaliaria pontos de todas as series com outra versao; informe os pontos")
        pts = all_points() if a.all else [p.resolve() for p in a.points]
        if not pts:
            ap.error("informe pontos ou use --all")
        for pt in pts:
            try:
                evaluate(pt, a.force, a.check_legacy)
            except Exception as e:
                print(f"[{pt}] ERRO: {e}")
