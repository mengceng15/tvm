'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import time
import mxnet as mx
import warnings

# from torch._C import T
warnings.filterwarnings("ignore")
# from mxnet.gluon.model_zoo.vision import *
import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay
import tvm.contrib.graph_executor as runtime
import numpy as np
from tvm.relay.testing import *
import os
from tvm.contrib import utils
from tvm.relay import nn

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

def update_lib(lib):
    # Include the path of src/runtime/contrib/dnnl/dnnl.cc
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    # source_dir = os.path.join(test_dir, "..", "..", "..")
    # contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")
    source_dir = os.path.join(test_dir, "..", "tvm")
    contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

    # Setup the gcc flag to compile DNNL code.
    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
    tmp_path = utils.tempdir()
    lib_name = 'lib.so'
    lib_path = tmp_path.relpath(lib_name)

    # The generated C code with DNNL APIs is compiled to a binary lib.so.
    lib.export_library(lib_path, fcompile=False, **kwargs)

    # Load the lib.so back to a runtime module.
    lib = tvm.runtime.load_module(lib_path)
    return lib

def example():
    x = relay.var("x", relay.TensorType((20, 30), "float32"))
    y = relay.var("y", relay.TensorType((30, 40), "float32"))
    z = relay.var("z", relay.TensorType((40, 10), "float32"))
    matmul0 = nn.matmul(x, y)
    matmul1 = nn.matmul(matmul0, z)
    # res = nn.relu(matmul1)
    return relay.Function([x, y, z], matmul1)

def benchmark(batch_size=1, batches=10, warmup=2):
    ctx = tvm.cpu()

    f = example()
    mod = tvm.IRModule.from_expr(f)
    # print(mod['main'].astext(show_meta_data=False))

    mod = relay.transform.AnnotateTarget(["dnnl"])(mod) # Output: Figure 2
    mod = relay.transform.MergeCompilerRegions()(mod) # Output: Figure 3
    mod = relay.transform.PartitionGraph()(mod) # Output: Figure 4
    # seq = tvm.transform.Sequential(
    #     [
    #         relay.transform.RemoveUnusedFunctions(),
    #         tvm.transform.PrintIR(),
    #         relay.transform.AlterOpLayout(),
    #         tvm.transform.PrintIR(),
    #         relay.transform.FoldConstant(),
    #         tvm.transform.PrintIR(),
    #         relay.transform.AnnotateTarget("dnnl"),
    #         tvm.transform.PrintIR(),
    #         relay.transform.MergeCompilerRegions(),
    #         tvm.transform.PrintIR(),
    #     ]
    # )

    # with tvm.transform.PassContext(opt_level=3):
    #     with tvm.target.Target("llvm"):
    #         mod = seq(mod)
    # print(mod['main'].astext(show_meta_data=False))

    # with relay.build_config(opt_level=3):
    #     graph, lib, params = relay.build(mod, target, params=params)

    json, lib, params = relay.build(mod, "llvm")
    rt_mod = tvm.contrib.graph_executor.create(json, lib, ctx)#Create a runtime executor module given a graph and module.

    datax = np.random.uniform(size=(20, 30))
    datay = np.random.uniform(size=(30, 40))
    dataz = np.random.uniform(size=(40, 10))

    rt_mod.set_input("x", tvm.nd.array(datax.astype("float32")))
    rt_mod.set_input("y", tvm.nd.array(datay.astype("float32")))
    rt_mod.set_input("z", tvm.nd.array(dataz.astype("float32")))
    rt_mod.run()
    tvm_output = rt_mod.get_output(0)
    print(tvm_output)
    print(np.matmul(np.matmul(datax, datay), dataz))

benchmark(batch_size=1)