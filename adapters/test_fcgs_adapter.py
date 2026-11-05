"""Teste das partes do adapter que não precisam de GPU nem do FCGS instalado."""
import importlib.util, sys, tempfile, types
from pathlib import Path
import numpy as np
from plyfile import PlyData

# --- funções copiadas literalmente de FCGS/model/gpcc_utils.py (commit 31e59a46)
GPCC = '''
import numpy as np
VOXELIZE_SCALE_FACTOR = 16
def voxelize(means: np.ndarray) -> tuple:
    means_min, means_max = means.min(axis=0), means.max(axis=0)
    voxelized_means = (means - means_min) / (means_max - means_min)  # normalize to [0, 1]
    voxelized_means = np.round(voxelized_means * (2 ** VOXELIZE_SCALE_FACTOR - 1))
    return voxelized_means, means_min, means_max
def devoxelize(voxelized_means, means_min, means_max):
    voxelized_means = voxelized_means.astype(np.float32)
    means_min = means_min.astype(np.float32)
    means_max = means_max.astype(np.float32)
    means = voxelized_means / (2 ** VOXELIZE_SCALE_FACTOR - 1) * (means_max - means_min) + means_min
    return means
def sorted_voxels(voxelized_means, other_params = None):
    indices_sorted = np.argsort(voxelized_means @ np.power(voxelized_means.max() + 1, np.arange(voxelized_means.shape[1])), axis=0)
    voxelized_means = voxelized_means[indices_sorted]
    if other_params is None:
        return voxelized_means
    other_params = other_params[indices_sorted]
    return voxelized_means, other_params
'''
tmp = Path(tempfile.mkdtemp())
(tmp / 'model').mkdir()
(tmp / 'model' / '__init__.py').write_text('')
(tmp / 'model' / 'gpcc_utils.py').write_text(GPCC)
sys.path.insert(0, str(tmp))
from model.gpcc_utils import voxelize, devoxelize, sorted_voxels

# --- torch: usa o real se houver; senão um substituto determinístico (só a lógica de ordem importa)
try:
    import torch
except ImportError:
    torch = types.ModuleType('torch')
    _st = {'seed': 0}
    torch.manual_seed = lambda s: _st.update(seed=s)
    class _P:
        def __init__(self, a): self.a = a
        def numpy(self): return self.a
    torch.randperm = lambda n: _P(np.random.default_rng(_st['seed']).permutation(n))
    sys.modules['torch'] = torch

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name('fcgs.py'))
spec = importlib.util.spec_from_file_location('ad', ADAPTER)
ad = importlib.util.module_from_spec(spec); spec.loader.exec_module(ad)
ad.FCGS_DIR = tmp

rng = np.random.default_rng(0)
N = 20000
xyz = rng.uniform(-3, 5, size=(N, 3)).astype(np.float32)
# cria duplicados de voxel: 300 Gaussianos copiam a posição de outro com jitter << meio passo
dup = rng.choice(N, 300, replace=False); tgt = rng.choice(N, 300, replace=False)
step = (xyz.max(0) - xyz.min(0)) / (2 ** 16 - 1)
xyz[dup] = xyz[tgt] + (rng.uniform(-0.1, 0.1, (300, 3)) * step).astype(np.float32)
names = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + ad.CANON[3:]
src = np.zeros(N, dtype=[(n, 'f4') for n in names])
src['x'], src['y'], src['z'] = xyz.T
for n in ad.CANON[3:]:
    src[n] = rng.normal(size=N).astype(np.float32)

# --- simula encoder e decoder do FCGS
vox, mn, mx = voxelize(xyz)
sv, order = sorted_voxels(vox, np.arange(N))
torch.manual_seed(1); perm = torch.randperm(N).numpy()
enc_src = order[perm]                                   # slot k do encoder <- Gaussiano enc_src[k]
dec_vox = sv[rng.permutation(N)]                        # G-PCC devolve fora de ordem
dsv = sorted_voxels(dec_vox)
torch.manual_seed(1); perm2 = torch.randperm(N).numpy()
pos = devoxelize(dsv[perm2], mn, mx)
dec = np.zeros(N, dtype=src.dtype)
dec['x'], dec['y'], dec['z'] = pos.T
for n in ad.CANON[3:]:
    dec[n] = src[n][enc_src]                            # atributos sem perda, na ordem do encoder
noise = 0.01
for i in range(3): dec[f'scale_{i}'] += noise

idx = ad.encoder_source_index(xyz)
assert np.array_equal(idx, enc_src), 'correspondência errada'
g = ad.geometry(src, dec, idx)
print('posição máx/meio passo:', round(g['position']['max_over_bound'], 4), '| com f32:', round(g['position']['max_over_bound_plus_f32'], 4))
assert g['position']['max_over_bound_plus_f32'] <= 1.0
assert abs(g['scale_log']['max'] - noise) < 1e-4 and abs(g['scale_log']['rmse'] - noise) < 1e-4
assert g['rotation_deg']['max'] < 1e-3, g['rotation_deg']
# rotação: q e -q dão 0°; 90° em torno de um eixo dá 90°
q = np.zeros(1, dtype=src.dtype); q['rot_0'] = 1
q2 = q.copy(); q2['rot_0'] = -1
h = np.sqrt(0.5); qa = q.copy(); qa['rot_0'] = h; qa['rot_1'] = h
for d in (q, qa):
    d['x'] = 0; d['y'] = 0; d['z'] = 0
two = np.concatenate([q, q]); two['x'][1] = 1; two['y'][1] = 1; two['z'][1] = 1
twoa = np.concatenate([qa, qa]); twoa['x'][1] = 1; twoa['y'][1] = 1; twoa['z'][1] = 1
ang = ad.geometry(two, twoa, np.array([0, 1]))['rotation_deg']['max']
assert abs(ang - 90) < 1e-6, ang
twom = two.copy(); twom['rot_0'] = -1
assert ad.geometry(two, twom, np.array([0, 1]))['rotation_deg']['max'] < 1e-6
print('rotação 90° e q/-q: ok')

# correspondência ingênua (sem reproduzir a ordem) precisa falhar: prova que o teste discrimina
bad = ad.geometry(src, dec, np.arange(N))['position']['max_over_bound']
assert bad > 1e3, bad
print('ordem ingênua falha como esperado (máx/limite =', round(bad), ')')

# PLY canônico de 59
out = tmp / 'c.ply'
ad.write_canonical(dec, out)
r = PlyData.read(str(out))['vertex'].data
assert list(r.dtype.names) == ad.CANON and len(r) == N
assert all(np.array_equal(r[c], dec[c]) for c in ad.CANON)
print('PLY de 59 propriedades: ok')
# cabeçalho de informação lateral: 3 B, ida e volta, e recusa estrutura incoerente
for l in ad.LAMBDAS:
    bd = tmp / f'bits_{l}'
    (bd / str(float(l)) / '0').mkdir(parents=True)
    assert ad.write_header(bd, l, 2) == 3 and (bd / ad.HEADER).stat().st_size == 3
    assert ad.read_header(bd) == (l, 2, 1)
(tmp / f'bits_1e-4' / '0.0001' / '1').mkdir()
try:
    ad.read_header(tmp / 'bits_1e-4'); raise SystemExit('cabeçalho incoerente não foi recusado')
except AssertionError:
    pass
print('cabeçalho de 3 B: ok')
print('TODOS OS TESTES PASSARAM')
