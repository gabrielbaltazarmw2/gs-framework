# Variáveis para compilar as extensões CUDA do FCGS. Usar com: conda activate fcgs && source este_arquivo
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export NVCC_CCBIN=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
