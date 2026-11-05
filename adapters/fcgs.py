#!/usr/bin/env python
"""Adapter do FCGS para o contrato de saída do framework (T3, passo 6).

Para cada λ: encode e decode com os scripts oficiais do FCGS (commit 31e59a46 + patches),
e grava o ponto em <results>/fcgs/<cena>/<frame>/lmd<λ>/ com:
  bitstream/      árvore <λ>/<passo>/ exatamente como o decoder a lê (T3 passo 7)
                  + fcgs_header.bin (3 B: índice de λ, chunk_div, número de passos). O wrapper
                  do decoder tira os parâmetros só desse arquivo, então não há informação
                  fora de banda (N420) e os 3 B entram na taxa pela regra única (5/11).
  decoded.ply     59 propriedades, ordem canônica (sem as normais zeradas do FCGS)
  manifest.json   contrato + codec_params + env + input
  geometry.json   erro de posição, escala e rotação contra a fonte (decisão C1, 4/11)
  run.json        registro bruto da execução (tempo, VRAM, bytes); não é resultado da T7

Roda no env `fcgs` (precisa de torch, numpy, plyfile e do tmc3 no PATH).
Uso:  python adapter_fcgs.py --scene breakdance_untracked --frame frame_0000 --lmd 1e-4
      python adapter_fcgs.py --scene breakdance_untracked --frame frame_0000 --lmd all
"""
import argparse, glob, hashlib, importlib.metadata, json, os, platform, re, resource
import shutil, struct, subprocess, sys, time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

HOME = Path.home()
FCGS_DIR = HOME / 'mpeg/methods/fcgs'
FRAMEWORK = HOME / 'mpeg/framework'
LAMBDAS = ['1e-4', '2e-4', '4e-4', '8e-4', '16e-4']
PATCHES = ['fcgs_31e59a46.patch', 'fcgs_31e59a46_chunk_div.patch']
CANON = (['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(45)]
         + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])
assert len(CANON) == 59
HEADER = 'fcgs_header.bin'
HEADER_FMT = '<BBB'   # índice de λ em LAMBDAS, chunk_div, número de passos


# ------------------------------------------- cabeçalho de informação lateral
def write_header(bits_dir, lmd, chunk_div):
    steps = [d for d in (bits_dir / str(float(lmd))).iterdir() if d.is_dir()]
    data = struct.pack(HEADER_FMT, LAMBDAS.index(lmd), chunk_div, len(steps))
    (bits_dir / HEADER).write_bytes(data)
    return len(data)


def read_header(bits_dir):
    i, chunk_div, n_steps = struct.unpack(HEADER_FMT, (bits_dir / HEADER).read_bytes())
    lmd = LAMBDAS[i]
    found = [d for d in (bits_dir / str(float(lmd))).iterdir() if d.is_dir()]
    assert len(found) == n_steps, f'cabeçalho diz {n_steps} passos, a pasta tem {len(found)}'
    return lmd, chunk_div, n_steps


def decode_bitstream(bits_dir, ply_out, log_path):
    """Wrapper do decoder: os parâmetros vêm só do cabeçalho dentro do bitstream."""
    lmd, chunk_div, _ = read_header(bits_dir)
    cmd = [sys.executable, 'decode_single_scene.py', '--lmd', lmd, '--chunk_div', str(chunk_div),
           '--bit_path_from', str(bits_dir), '--ply_path_to', str(ply_out)]
    return timed(cmd, FCGS_DIR, log_path)


# ---------------------------------------------------------------- utilidades
def file_hash(p, algo='sha256'):
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def out(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_state():
    q = out('nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader,nounits')
    name, drv, used, tot = [x.strip() for x in q.splitlines()[0].split(',')]
    return dict(gpu=name, driver=drv, vram_used_mib=int(used), vram_total_mib=int(tot))


ALLOC_CONF = 'expandable_segments:True'   # sem isso, na 4070 o pico encosta na VRAM e o encode
                                          # fica ~24x mais lento (transbordo provável); bitstream idêntico (5/11)
VRAM_WARN = 0.95                          # pico acima disso: tempo marcado como suspeito


def timed(cmd, cwd, log_path):
    """Cronômetro só em volta da chamada do codec. CPU inclui os filhos (tmc3).
    Amostra a VRAM a cada 200 ms para registrar o pico."""
    import threading
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = ALLOC_CONF
    smi = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits',
                            '-lms', '200'], stdout=subprocess.PIPE, text=True)
    samples = []
    def reader():
        for line in smi.stdout:
            try:
                u, t = (int(x) for x in line.split(','))
                samples.append((u, t))
            except ValueError:
                pass
    th = threading.Thread(target=reader, daemon=True); th.start()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    with open(log_path, 'w') as f:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    t1 = time.perf_counter()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    smi.terminate(); smi.wait(); th.join(timeout=2)   # esperado depois de r1: nvidia-smi fica fora do delta
    peak = max((u for u, _ in samples), default=None)
    total = samples[0][1] if samples else None
    return rc, dict(wall_s=round(t1 - t0, 3),
                    cpu_user_s=round(r1.ru_utime - r0.ru_utime, 3),
                    cpu_sys_s=round(r1.ru_stime - r0.ru_stime, 3),
                    vram_peak_mib=peak,
                    vram_near_limit=bool(peak and total and peak > VRAM_WARN * total),
                    alloc_conf=ALLOC_CONF)


def find_md5(ply):
    """Procura o md5 oficial nos .md5 da pasta da sequência (formato '<hash> *plys/<frame>.ply')."""
    rel = f'{ply.parent.name}/{ply.name}'
    for m in sorted(glob.glob(str(ply.parent.parent / '*.md5'))):
        for line in open(m):
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip('*') in (rel, ply.name):
                return parts[0]
    return None


# ------------------------------------------------- ordem do encoder do FCGS
def step_size(n):
    """Igual a encode_single_scene.py: passos de 1e6 Gaussianos (1,1e6 se 1e6 < n < 1,1e6)."""
    return 1_100_000 if 1_000_000 < n < 1_100_000 else 1_000_000


def encoder_source_index(xyz_f32):
    """Índice na fonte de cada Gaussiano decodificado, reproduzindo o encoder na mesma máquina.

    Encoder (FCGS_model.compress): sorted_orig_voxels (voxelize + sorted_voxels) e depois
    torch.manual_seed(1); randperm. O decoder ordena os voxels decodificados do mesmo jeito
    e aplica a mesma randperm, então o slot k do decodificado recebe os atributos do slot
    sorted[perm[k]] do encoder. Usa as próprias funções do FCGS para ser exato.
    """
    import torch
    sys.path.insert(0, str(FCGS_DIR))
    from model.gpcc_utils import voxelize, sorted_voxels
    n = xyz_f32.shape[0]
    P = step_size(n)
    idx_all = []
    for s in range(int(np.ceil(n / P))):
        lo = s * P
        x = xyz_f32[lo:lo + P]
        vox, _, _ = voxelize(means=x)
        _, order = sorted_voxels(voxelized_means=vox, other_params=np.arange(x.shape[0]))
        torch.manual_seed(1)
        perm = torch.randperm(x.shape[0]).numpy()
        idx_all.append(lo + order[perm])
    return np.concatenate(idx_all)


# ------------------------------------------------------------- geometria
def geometry(src, dec, src_idx):
    """Erro de posição (por eixo), escala (espaço log do PLY) e rotação (ângulo entre
    quaternions normalizados, 2·acos|<q1,q2>|), máximo e RMSE. Decisão de 4/11."""
    s = src[src_idx]
    pos_s = np.stack([s['x'], s['y'], s['z']], 1).astype(np.float64)
    pos_d = np.stack([dec['x'], dec['y'], dec['z']], 1).astype(np.float64)
    ep = np.abs(pos_d - pos_s)
    # limite analítico: meio passo da grade de 2^16 - 1 níveis sobre a caixa de cada passo
    xyz_all = np.stack([src['x'], src['y'], src['z']], 1).astype(np.float64)
    n, P = len(src), step_size(len(src))
    bound = np.zeros(3)
    tol = np.zeros(3)   # arredondamento de float32 na voxelização e na volta
    for k in range(int(np.ceil(n / P))):
        b = xyz_all[k * P:(k + 1) * P]
        bound = np.maximum(bound, (b.max(0) - b.min(0)) / (2 * (2 ** 16 - 1)))
        tol = np.maximum(tol, 4 * np.finfo(np.float32).eps * np.maximum(np.abs(b.min(0)), np.abs(b.max(0))))
    es = np.abs(np.stack([dec[f'scale_{i}'] for i in range(3)], 1).astype(np.float64)
                - np.stack([s[f'scale_{i}'] for i in range(3)], 1).astype(np.float64))
    qs = np.stack([s[f'rot_{i}'] for i in range(4)], 1).astype(np.float64)
    qd = np.stack([dec[f'rot_{i}'] for i in range(4)], 1).astype(np.float64)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    qd /= np.linalg.norm(qd, axis=1, keepdims=True)
    ang = np.degrees(2 * np.arccos(np.clip(np.abs((qs * qd).sum(1)), 0.0, 1.0)))
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    return dict(
        n=int(len(dec)),
        position=dict(max_per_axis=ep.max(0).tolist(), rmse_per_axis=[rms(ep[:, i]) for i in range(3)],
                      half_step_bound_per_axis=bound.tolist(),
                      float32_tolerance_per_axis=tol.tolist(),
                      max_over_bound=float((ep.max(0) / bound).max()),
                      max_over_bound_plus_f32=float((ep.max(0) / (bound + tol)).max())),
        scale_log=dict(max=float(es.max()), rmse=rms(es)),
        rotation_deg=dict(max=float(ang.max()), rmse=rms(ang)),
        correspondence='encoder order reproduced (voxelize + sorted_voxels + randperm seed 1), same machine',
    )


def write_canonical(dec, path):
    """PLY de 59 propriedades na ordem canônica (descarta nx, ny, nz)."""
    missing = [c for c in CANON if c not in dec.dtype.names]
    assert not missing, f'propriedades ausentes no PLY do FCGS: {missing}'
    arr = np.empty(len(dec), dtype=[(c, 'f4') for c in CANON])
    for c in CANON:
        arr[c] = dec[c]
    PlyData([PlyElement.describe(arr, 'vertex')], text=False, byte_order='<').write(str(path))


# ------------------------------------------------------------ ambiente
def framework_state():
    """Commit do framework e se a cópia está suja (regra de Git de 5/11). Suja = arquivo
    versionado modificado, ou o próprio adapter fora do git ou modificado."""
    commit = out('git rev-parse --short HEAD', cwd=FRAMEWORK)
    tracked = out('git status --porcelain --untracked-files=no', cwd=FRAMEWORK) or ''
    me = os.path.relpath(Path(__file__).resolve(), FRAMEWORK)
    mine = out(f'git status --porcelain -- "{me}"', cwd=FRAMEWORK) or ''
    dirty = sorted({l[3:] for l in (tracked + '\n' + mine).splitlines() if l.strip()})
    return commit, bool(dirty), dirty


def environment():
    import torch
    commit = out('git rev-parse HEAD', cwd=FCGS_DIR)
    fw_commit, fw_dirty, fw_dirty_files = framework_state()
    diff = subprocess.run('git diff', shell=True, cwd=FCGS_DIR, capture_output=True).stdout
    tmc3 = out('tmc3 --help 2>&1 | head -1')
    env = dict(machine=platform.node(), python=platform.python_version(),
               torch=torch.__version__, torch_cuda=torch.version.cuda,
               numpy=np.__version__, plyfile=importlib.metadata.version('plyfile'),
               tmc3=tmc3, fcgs_commit=commit,
               framework_commit=fw_commit, framework_dirty=fw_dirty,
               framework_dirty_files=fw_dirty_files,
               fcgs_worktree_diff_sha256=hashlib.sha256(diff).hexdigest(),
               patches={p: file_hash(FRAMEWORK / 'patches' / p) for p in PATCHES
                        if (FRAMEWORK / 'patches' / p).exists()})
    g = gpu_state()
    env.update(gpu=g['gpu'], driver=g['driver'], vram_total_mib=g['vram_total_mib'])
    return commit, env


# ------------------------------------------------------------- um ponto
def run_point(a, lmd, commit, env):
    point = a.results_root / 'fcgs' / a.scene / a.frame / f'lmd{lmd}'
    if point.exists() and not a.force:
        print(f'[{lmd}] já existe, pulando (use --force): {point}')
        return
    work = a.work_root / a.scene / a.frame / f'lmd{lmd}'
    shutil.rmtree(work, ignore_errors=True)
    (work / 'bits').mkdir(parents=True)

    g0 = gpu_state()
    if g0['vram_used_mib'] > a.max_base_vram:
        sys.exit(f'[{lmd}] VRAM de base {g0["vram_used_mib"]} MiB acima de {a.max_base_vram} (regra da T6). Abortado.')

    py = sys.executable
    enc_cmd = [py, 'encode_single_scene.py', '--lmd', lmd, '--chunk_div', str(a.chunk_div),
               '--ply_path_from', str(a.input), '--bit_path_to', str(work / 'bits'), '--determ', '1']
    rc, t_enc = timed(enc_cmd, FCGS_DIR, work / 'encode.log')
    if rc != 0:
        sys.exit(f'[{lmd}] encode falhou (exit {rc}); ver {work / "encode.log"}')
    m = re.search(r'^time:\s*([0-9.eE+-]+)', (work / 'encode.log').read_text(), re.M)
    t_enc['fcgs_internal_compress_s'] = float(m.group(1)) if m else None

    header_bytes = write_header(work / 'bits', lmd, a.chunk_div)
    raw = work / 'decoded_fcgs62.ply'
    rc, t_dec = decode_bitstream(work / 'bits', raw, work / 'decode.log')   # só o cabeçalho, sem --lmd
    if rc != 0:
        sys.exit(f'[{lmd}] decode falhou (exit {rc}); ver {work / "decode.log"}')

    files = sorted(p for p in (work / 'bits').rglob('*') if p.is_file())
    rate_bytes = sum(p.stat().st_size for p in files)

    src = PlyData.read(str(a.input))['vertex'].data
    dec = PlyData.read(str(raw))['vertex'].data
    assert len(dec) == len(src), f'N decodificado {len(dec)} != fonte {len(src)}'
    xyz = np.stack([src['x'], src['y'], src['z']], 1).astype(np.float32)
    geo = geometry(src, dec, encoder_source_index(xyz))

    # escreve o ponto em pasta temporária e renomeia no fim (ponto parcial nunca fica em results/)
    tmp = point.with_name(point.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copytree(work / 'bits', tmp / 'bitstream')
    write_canonical(dec, tmp / 'decoded.ply')
    manifest = dict(method='fcgs', method_version=commit, scene=a.scene, frame=a.frame,
                    rate_param=dict(type='lambda', value=lmd), conda_env='fcgs',
                    rate_source='measured_bitstream',
                    codec_params=dict(chunk_div=a.chunk_div, determ=1,
                                      side_info=f'{HEADER} ({header_bytes} B) dentro de bitstream/'),
                    env=env,
                    input=dict(path=str(a.input), md5=a.input_md5, md5_checked=a.md5_checked))
    (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (tmp / 'geometry.json').write_text(json.dumps(geo, indent=2))
    run = dict(encode=t_enc, decode=t_dec, vram_base_mib=g0['vram_used_mib'],
               rate_bytes=rate_bytes, n_files=len(files), header_bytes=header_bytes,
               decoded_fcgs62_sha256=file_hash(raw),
               decoded_canonical59_sha256=file_hash(tmp / 'decoded.ply'),
               note='registro bruto de uma execução; o protocolo de tempo é da T7 (timing.json)')
    (tmp / 'run.json').write_text(json.dumps(run, indent=2))
    if point.exists():
        shutil.rmtree(point)
    tmp.rename(point)
    if not a.keep_work:
        shutil.rmtree(work)
    for nome, t in (('encode', t_enc), ('decode', t_dec)):
        if t['vram_near_limit']:
            print(f'[{lmd}] AVISO: pico de VRAM no {nome} = {t["vram_peak_mib"]} MiB (> {VRAM_WARN:.0%}); '
                  f'tempo suspeito (possível transbordo para a RAM do sistema)')
    if env['framework_dirty']:
        print(f'[{lmd}] AVISO: framework com cópia suja ({env["framework_dirty_files"]}); ponto vale só como teste')
    print(f'[{lmd}] ok: {len(files)} arquivos, {rate_bytes} B, encode {t_enc["wall_s"]} s, '
          f'decode {t_dec["wall_s"]} s, posição máx/(meio passo + f32) {geo["position"]["max_over_bound_plus_f32"]:.3f} -> {point}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--frame', required=True)
    p.add_argument('--lmd', required=True, help="'all' ou um de " + ', '.join(LAMBDAS))
    p.add_argument('--input', type=Path, help='padrão: ~/mpeg/datasets/<cena>/plys/<frame>.ply')
    p.add_argument('--md5', help='md5 oficial da entrada (padrão: procurar nos .md5 da sequência)')
    p.add_argument('--chunk_div', type=int, default=2)
    p.add_argument('--results_root', type=Path, default=HOME / 'mpeg/results')
    p.add_argument('--work_root', type=Path, default=HOME / 'mpeg/scratch/fcgs_adapter')
    p.add_argument('--max_base_vram', type=int, default=1000)
    p.add_argument('--force', action='store_true')
    p.add_argument('--keep_work', action='store_true', help='mantém a pasta de trabalho (inclui o PLY de 62)')
    a = p.parse_args()
    lmds = LAMBDAS if a.lmd == 'all' else [a.lmd]
    assert all(l in LAMBDAS for l in lmds), f'λ inválido: {a.lmd}'
    a.input = (a.input or HOME / f'mpeg/datasets/{a.scene}/plys/{a.frame}.ply').resolve()
    assert a.input.exists(), f'entrada não encontrada: {a.input}'
    assert shutil.which('tmc3'), 'tmc3 não está no PATH (ative o env fcgs)'

    ref = a.md5 or find_md5(a.input)
    a.input_md5 = file_hash(a.input, 'md5')
    a.md5_checked = ref is not None
    if ref and ref != a.input_md5:
        sys.exit(f'md5 da entrada {a.input_md5} difere do oficial {ref}')
    print(f'entrada: {a.input} | md5 {a.input_md5} | conferido: {a.md5_checked}')

    commit, env = environment()
    for l in lmds:
        run_point(a, l, commit, env)


if __name__ == '__main__':
    main()
