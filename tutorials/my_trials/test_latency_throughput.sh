#!/bin/bash
export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

numactl --physcpubind=0-27 --membind=0 \
python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py --batch-size=1
