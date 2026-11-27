#!/usr/bin/env python
"""T16 p3, diagnóstico só de CPU: as posições que o decoder reconstrói do xyz_gpcc.bin são,
bit a bit, as mesmas que o encoder usou para montar os contextos?

Encoder (FCGS_model.compress, l.342): sorted_orig_voxels -> voxeliza com (min, max) do PLY, ordena,
devoxeliza -> g_xyz, que alimenta normalize_xyz e as grades de contexto. Depois compress_gaussian_params
revoxeliza g_xyz com um (min, max) NOVO, calculado de g_xyz, e grava esse par no cabeçalho.
Decoder (decomprss, l.616): G-PCC -> ordena -> devoxeliza com o (min, max) do cabeçalho.
Se os dois caminhos não derem os mesmos floats, os contextos diferem entre encoder e decoder.

Usa as funções do próprio FCGS (model/gpcc_utils.py), que são numpy; nada vai para a GPU.

Uso:
  python t16_pos_check.py --fcgs ~/mpeg/methods/fcgs --ply <PLY de entrada do encoder> \
      --bits <pasta bitstream de um ponto> [--tmc3 tmc3]
  python t16_pos_check.py --fcgs ~/mpeg/methods/fcgs --ply <PLY> --simulate   # sem G-PCC (autoteste)
"""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fcgs", type=Path, required=True)
ap.add_argument("--ply", type=Path, required=True)
ap.add_argument("--bits", type=Path, help="pasta bitstream/ de um ponto (procura xyz_gpcc.bin dentro)")
ap.add_argument("--tmc3", default="tmc3")
ap.add_argument("--simulate", action="store_true", help="em vez do G-PCC, embaralha os inteiros (autoteste)")
a = ap.parse_args()

sys.path.insert(0, str(a.fcgs.expanduser()))
from model.gpcc_utils import voxelize, devoxelize, sorted_voxels, read_ply_geo_bin, read_binary_data  # noqa: E402
from plyfile import PlyData  # noqa: E402

# --- encoder (igual a sorted_orig_voxels + compress_gaussian_params) ---
v = PlyData.read(str(a.ply))["vertex"].data
x = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
N = len(x)
vox, mn, mx = voxelize(means=x)
v1 = sorted_voxels(voxelized_means=vox, other_params=None)
g_enc = devoxelize(voxelized_means=v1, means_min=mn, means_max=mx)          # o que alimenta os contextos
v2, mn2, mx2 = voxelize(means=g_enc)                                       # o que vai para o G-PCC
v2 = sorted_voxels(voxelized_means=v2, other_params=None)

# --- decoder ---
if a.simulate:
    dec_int = v2[np.random.default_rng(0).permutation(N)]
    hmn, hmx = mn2.astype(np.float32), mx2.astype(np.float32)
    fonte = "simulado (inteiros do encoder embaralhados)"
else:
    bins = sorted(a.bits.expanduser().rglob("xyz_gpcc.bin"))
    assert len(bins) == 1, f"esperava 1 xyz_gpcc.bin em {a.bits}, achei {bins}"
    with tempfile.TemporaryDirectory() as td, open(bins[0], "rb") as f:
        head = np.frombuffer(f.read(24), dtype=np.float32)
        hmn, hmx = head[:3].copy(), head[3:].copy()
        payload = os.path.join(td, "compressed.bin"); read_binary_data(dst_bin_path=payload, src_file_handle=f)
        rec = os.path.join(td, "rec.ply")
        r = subprocess.run([a.tmc3, "--mode=1", "--outputBinaryPly=1", f"--compressedStreamPath={payload}",
                            f"--reconstructedDataPath={rec}"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        dec_int = read_ply_geo_bin(ply_path=rec).astype(np.float32)
    fonte = str(bins[0])
dec_sorted = sorted_voxels(voxelized_means=dec_int)
g_dec = devoxelize(voxelized_means=dec_sorted, means_min=hmn, means_max=hmx)

def ulps(p, q):
    return (p.astype(np.float32).view(np.int32).astype(np.int64) - q.astype(np.float32).view(np.int32).astype(np.int64)).tolist()

n_dup = N - len(np.unique(vox, axis=0))
print(f"PLY: {a.ply}  N={N}  voxels duplicados={n_dup}")
print(f"decoder: {fonte}")
print(f"caixa do PLY      min={mn.tolist()} max={mx.tolist()}")
step = (mx - mn).astype(np.float64) / (2 ** 16 - 1)
ulp = np.spacing(np.maximum(np.abs(mn), np.abs(mx)).astype(np.float32)).astype(np.float64)
print(f"passo do voxel / ulp do float32 por eixo: {np.round(step / ulp, 2).tolist()}  (perto de 1 = risco: o float32 mal representa a grade)")
print(f"caixa revoxelizada (cabeçalho) Δmin em ulps={ulps(mn2, mn)} Δmax em ulps={ulps(mx2, mx)}")
if not a.simulate:
    print(f"cabeçalho lido == (min, max) revoxelizado: {np.array_equal(hmn, mn2.astype(np.float32)) and np.array_equal(hmx, mx2.astype(np.float32))}")
print(f"inteiros: 1ª voxelização == revoxelização: {np.array_equal(v1, v2)}  (linhas diferentes: {int((v1 != v2).any(1).sum())})")
print(f"inteiros: G-PCC decodificado == enviado ao G-PCC: {np.array_equal(dec_sorted, v2)}")
eq = np.array_equal(g_dec, g_enc)
nd = int((g_dec != g_enc).any(1).sum())
print(f"POSIÇÕES decoder == encoder bit a bit: {eq}  (linhas diferentes: {nd} de {N}; máx |Δ| = {float(np.abs(g_dec - g_enc).max()):.3e})")
print("VEREDITO:", "posições idênticas -> este mecanismo NÃO explica a falha" if eq
      else "posições diferentes -> contextos diferem entre encoder e decoder (candidato a causa)")
