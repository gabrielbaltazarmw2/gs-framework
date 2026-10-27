#!/usr/bin/env bash
# Acrescenta noexcept (true) a rsqrt, rsqrtf, sinpi, sinpif, cospi e cospif em
# crt/math_functions.h do CUDA 12.8, que conflita com a glibc 2.43 do Ubuntu da workstation.
# Substitui o patch do header: o repositorio nao contem linhas do arquivo da NVIDIA.
# So muda assinatura de declaracao, sem efeito numerico. Idempotente.
# Uso: conda activate fcgs && bash envs/patch_cuda_math_functions.sh [caminho/math_functions.h]
set -euo pipefail
H="${1:-${CONDA_PREFIX:?ative o env fcgs ou passe o caminho}/targets/x86_64-linux/include/crt/math_functions.h}"
[ -f "$H" ] || { echo "nao encontrado: $H" >&2; exit 1; }
[ -f "$H.orig" ] || cp -p "$H" "$H.orig"
sed -i 's/^\(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ [a-z]\+ \+\(rsqrtf\?\|sinpif\?\|cospif\?\)([a-z]\+ x)\);$/\1 noexcept (true);/' "$H"
n=$(grep -cE '^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ [a-z]+ +(rsqrtf?|sinpif?|cospif?)\([a-z]+ x\) noexcept \(true\);$' "$H")
[ "$n" -eq 6 ] || { echo "esperava 6 declaracoes com noexcept, achei $n" >&2; exit 1; }
echo "ok: 6 declaracoes com noexcept (true) em $H"
