#!/bin/bash
export OMP_NUM_THREADS=4
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

for ((i=0; i<7; i++))
    do
        startCore=$[${i}*4]
        endCore=$[${startCore}+3]
        numactl --physcpubind=${startCore}-${endCore} --membind=0 \
        python /home2/zhangya9/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py&
    done
