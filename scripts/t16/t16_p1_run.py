#!/usr/bin/env python
"""T16 passo 1: encode e decode com os scripts oficiais do FCGS, fora do adapter, e comparação
com um ponto já gravado pelo adapter (adapters/fcgs.py, commit 5413f3c).

Todos os parâmetros vêm do manifest.json do ponto (λ, chunk_div, entrada do encoder e md5), para
reproduzir a chamada do adapter sem digitar nada à mão. Só escreve em --out (scratch).

Uso (env fcgs, de ~/mpeg/framework):
  python scripts/t16/t16_p1_run.py --point ~/mpeg/results/fcgs/<cena>/<frame>/lmd1e-4
  python scripts/t16/t16_p1_run.py --point ... --chunk_div 4     # só se o ponto for div 4 (senão aborta)
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t16_p1_check import digest, manifest as tree_manifest

HOME = Path.home()
FCGS_DIR = HOME / 'mpeg/methods/fcgs'          # igual ao adapter
RESULTS = HOME / 'mpeg/results'
HEADER = 'fcgs_header.bin'                      # 3 B que o adapter acrescenta ao bitstream
ALLOC_CONF = 'expandable_segments:True'         # igual ao adapter (bitstream idêntico, 5/11)


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True)
    return r.stdout


def vram_used():
    q = sh('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits').decode()
    u, t = (int(x) for x in q.splitlines()[0].split(','))
    return u, t


def run(cmd, log, env):
    t0 = time.perf_counter()
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, cwd=FCGS_DIR, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f'falhou (exit {rc}): {" ".join(map(str, cmd))}\nver {log}')
    return round(time.perf_counter() - t0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--point', type=Path, required=True)
    p.add_argument('--chunk_div', type=int, help='padrão: o do ponto; diferente do ponto aborta')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    pt = a.point.expanduser().resolve()
    man = json.loads((pt / 'manifest.json').read_text())
    rj = json.loads((pt / 'run.json').read_text())

    lmd = man['rate_param']['value']
    div_pt = man['codec_params']['chunk_div']
    div = a.chunk_div or div_pt
    if div != div_pt:
        sys.exit(f'chunk_div {div} ≠ {div_pt} do ponto: o chunk_size_list muda, então o bitstream muda por '
                 f'construção e a comparação não diz nada sobre a integração. Abortado.')
    enc = man.get('encoder_input') or man['input']
    enc_path, enc_md5 = Path(enc['path']), enc['md5']

    # 1) mesma entrada, mesmo FCGS, mesmos patches
    md5_now = digest(enc_path, 'md5')
    assert md5_now == enc_md5, f'md5 da entrada do encoder {md5_now} ≠ manifest {enc_md5}'
    commit = sh('git rev-parse HEAD', FCGS_DIR).decode().strip()
    diff_sha = hashlib.sha256(sh('git diff', FCGS_DIR)).hexdigest()
    assert commit == man['method_version'], f'FCGS em {commit}, ponto feito em {man["method_version"]}'
    assert diff_sha == man['env']['fcgs_worktree_diff_sha256'], 'patches do FCGS diferentes dos do ponto'

    out = (a.out or HOME / f'mpeg/scratch/t16/p1/{man["scene"]}_{man["frame"]}_lmd{lmd}_div{div}').expanduser().resolve()
    assert RESULTS.resolve() not in out.parents and out != RESULTS.resolve(), 'saída não pode ficar em results/'
    if out.exists():
        sys.exit(f'{out} já existe; apague ou use --out')
    (out / 'bits').mkdir(parents=True)

    # 2) mesmas chamadas do adapter (mesma ordem de argumentos), fora dele
    env = os.environ.copy(); env['PYTORCH_CUDA_ALLOC_CONF'] = ALLOC_CONF
    base, total = vram_used()
    py = sys.executable
    t_enc = run([py, 'encode_single_scene.py', '--lmd', lmd, '--chunk_div', str(div),
                 '--ply_path_from', str(enc_path), '--bit_path_to', str(out / 'bits'), '--determ', '1'],
                out / 'encode.log', env)
    t_dec = run([py, 'decode_single_scene.py', '--lmd', lmd, '--chunk_div', str(div),
                 '--bit_path_from', str(out / 'bits'), '--ply_path_to', str(out / 'decoded.ply')],
                out / 'decode.log', env)

    # 3) comparação
    A = {k: v for k, v in tree_manifest(pt / 'bitstream').items() if k != HEADER}
    B = tree_manifest(out / 'bits')
    keys = sorted(set(A) | set(B))
    diffs = [k for k in keys if A.get(k, (None, None))[1] != B.get(k, (None, None))[1]]
    bytes_A, bytes_B = sum(v[0] for v in A.values()), sum(v[0] for v in B.values())
    hdr = (pt / 'bitstream' / HEADER).stat().st_size
    dec_sha = digest(out / 'decoded.ply', 'sha256')
    bits_ok = not diffs and bytes_A == bytes_B and bytes_B + hdr == rj['rate_bytes']
    dec_ok = dec_sha == rj['decoded_fcgs62_sha256']

    rep = dict(point=str(pt), out=str(out), lmd=lmd, chunk_div=div,
               encoder_input=dict(path=str(enc_path), md5=md5_now, tratada=bool(man.get('encoder_input'))),
               fcgs_commit=commit, fcgs_diff_sha256=diff_sha,
               vram_base_mib=base, vram_total_mib=total, wall_s=dict(encode=t_enc, decode=t_dec),
               bitstream=dict(arquivos_adapter=len(A), arquivos_oficial=len(B),
                              bytes_adapter_sem_header=bytes_A, bytes_oficial=bytes_B,
                              header_bytes=hdr, rate_bytes_run_json=rj['rate_bytes'],
                              arquivos_diferentes=diffs, identico=bits_ok),
               decoded62=dict(sha256_adapter=rj['decoded_fcgs62_sha256'], sha256_oficial=dec_sha,
                              identico=dec_ok),
               veredito='integração descartada' if bits_ok and dec_ok else 'DIFERENTE: localizar')
    (out / 'report.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print('\nlista por arquivo:')
    for k in keys:
        sa, sb = A.get(k), B.get(k)
        print(f'  {k:32s} {sa[0] if sa else "—":>10} {sb[0] if sb else "—":>10}  '
              f'{"igual" if sa and sb and sa[1] == sb[1] else "DIFERENTE"}')


if __name__ == '__main__':
    main()
