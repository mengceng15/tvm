# BYOC with DNNL

## Requirement
Tested with byoc: mxnet==1.8.0.post0  
Tested with onednn: mxnet==2.0.0a0  

## Build
mkdir build  
cp cmake/config.cmake build  
cd build  
  
set(USE_LLVM ON)  
set(USE_MKLDNN ON)  
set(USE_DNNL_CODEGEN ON)  
set(USE_OPENEM gnu)  
  
export CXXFLAGS="-I$HOME/built_onednn/include"  
cmake .. -DEXTERN_LIBRARY_DNNL=$HOME/built_onednn/lib64/libdnnl.so -DMKLDNN_LIBRARY=$HOME/built_onednn/lib64/libmkldnn.so  
make -j  

## Benchmark
- Commands for opt_byoc, original byoc and onednnbs=1 / bs=128 28core
```bash
./tutorials/my_trials/test_latency_throughput.sh
```
```bash
OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 numactl --physcpubind=0-27 --membind=0 python benchmark_onednn.py
```

- Commands for multi-instance
```bash
./tutorials/my_trials/test_realtime.sh
```

