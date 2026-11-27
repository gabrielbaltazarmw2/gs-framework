#!/usr/bin/env python
"""T16 p3 (1b): confirma o mecanismo da dessincronização isolando as camadas suspeitas.

Remonta, a partir dos dumps do encoder (t16_roundtrip.py enc), o g_fea_enc_q que o encoder codificou
e aplica, com os pesos reais do checkpoint, cada camada em duas formas:
  - latdim_2_griddim_fea: sobre as N_fea linhas (encoder, l.418) contra o prefixo [k0:k_st] (decoder, l.684);
  - fea_channel_ctx (MLPs dos grupos 1 a 3): sobre N_fea linhas (encoder, l.438) contra a fatia
    [k_st:k_ed] com to_dec (decoder, l.694).
Mesmas entradas, só o número de linhas muda. Diferença bit a bit = a saída depende do número de linhas.

Uso (na raiz da árvore do FCGS): python t16_mechanism.py <dump_dir> --lmd 1e-4 [--nr 3]
"""
import argparse, os, sys
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dump", type=Path)
ap.add_argument("--lmd", type=float, required=True)
ap.add_argument("--nr", type=float, default=3)
a = ap.parse_args()
sys.path.insert(0, os.getcwd())
from model.FCGS_model import FCGS  # noqa: E402

E = a.dump / "enc"
# remonta g_fea_enc_q [N_fea, 256] na ordem do encoder (fatias sp0..sp3; grupos de 64 canais)
blocks, kn = [], [0]
for sp in range(4):
    cols = [torch.load(E / f"0_g_fea_enc_q_sp{sp}_ch{c}.b.pt")["sym"].view(-1, 64) for c in range(4)]
    blocks.append(torch.cat(cols, dim=1)); kn.append(kn[-1] + blocks[-1].shape[0])
q = torch.cat(blocks, 0).cuda()
print(f"N_fea={q.shape[0]}  kn={kn}")

CM = FCGS(Q=1, resolutions_list=[300, 400, 500], resolutions_list_3D=[70, 80, 90], norm_radius=a.nr).cuda()
CM.load_state_dict(torch.load(f"./checkpoints/checkpoint_{a.lmd}.pkl"), strict=True)
CM.eval()

def cmp(name, x, y):
    ne = (x != y).any(dim=1)
    n = int(ne.sum())
    if n == 0:
        print(f"{name:52s} idêntico ({x.shape[0]} linhas)")
    else:
        r = int(ne.nonzero()[0, 0])
        print(f"{name:52s} DIFERE em {n} de {x.shape[0]} linhas; 1ª linha {r}; máx |Δ| {float((x - y).abs().max()):.3e}")
    return n

tot = 0
with torch.no_grad():
    full = CM.latdim_2_griddim_fea(q)                       # encoder, l.418
    for l_sp in (1, 2, 3):
        pref = CM.latdim_2_griddim_fea(q[kn[0]:kn[l_sp]])  # decoder, l.684
        tot += cmp(f"latdim_2_griddim_fea  N_fea vs prefixo [0:{kn[l_sp]}]", full[kn[0]:kn[l_sp]], pref)
    m_full, s_full, p_full = CM.fea_channel_ctx.forward(q)  # encoder, l.438
    for l_sp in range(4):
        k_st, k_ed = kn[l_sp], kn[l_sp + 1]
        for l_ch in (1, 2, 3):
            c0, c1 = 64 * l_ch, 64 * l_ch + 64
            m, s, p = CM.fea_channel_ctx.forward(q[k_st:k_ed], to_dec=l_ch)   # decoder, l.694
            x = torch.cat([m_full[k_st:k_ed, c0:c1], s_full[k_st:k_ed, c0:c1], p_full[k_st:k_ed, c0:c1]], 1)
            y = torch.cat([m, s, p], 1)
            tot += cmp(f"fea_channel_ctx sp{l_sp} ch{l_ch}  N_fea vs fatia [{k_st}:{k_ed}]", x, y)
print("\nVEREDITO:", "a saída depende do número de linhas (mecanismo confirmado nesta GPU)" if tot
      else "nenhuma diferença: as camadas não dependem do número de linhas nesta GPU (mecanismo não confirmado)")
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
