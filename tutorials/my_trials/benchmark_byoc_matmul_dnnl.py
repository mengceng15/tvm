'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import time
import mxnet as mx
import warnings

import math

from tvm.relay.op.tensor import bitwise_and

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

def matmul_bias_example():
    x = relay.var("x", relay.TensorType((4, 6), "float32"))
    y = relay.var("y", relay.TensorType((6, 3), "float32"))
    z = relay.var("z", relay.TensorType((3, 2), "float32"))
    b = relay.var("b", relay.TensorType((4, 2), "float32"))
    matmul0 = nn.matmul(x, y)
    matmul1 = nn.matmul(matmul0, z)
    bias = relay.add(matmul1, b)

    return relay.Function([x, y, z, b], bias)

def matmul_bias_relu_example():
    x = relay.var("x", relay.TensorType((4, 6), "float32"))
    y = relay.var("y", relay.TensorType((6, 3), "float32"))
    z = relay.var("z", relay.TensorType((3, 2), "float32"))
    b = relay.var("b", relay.TensorType((4, 2), "float32"))
    matmul0 = nn.matmul(x, y)
    matmul1 = nn.matmul(matmul0, z)
    bias = relay.add(matmul1, b)
    # relu
    res = nn.relu(bias)

    return relay.Function([x, y, z, b], res)

def matmul_bias_gelu_example():
    x = relay.var("x", relay.TensorType((4, 6), "float32"))
    y = relay.var("y", relay.TensorType((6, 3), "float32"))
    z = relay.var("z", relay.TensorType((3, 2), "float32"))
    b = relay.var("b", relay.TensorType((4, 2), "float32"))
    matmul0 = nn.matmul(x, y)
    matmul1 = nn.matmul(matmul0, z)
    bias = relay.add(matmul1, b)
    # gelu
    const1 = relay.const(0.044715)
    const2 = relay.const(math.sqrt(2 / math.pi))
    gelu = relay.power(bias, relay.const(3, dtype="float32"))
    gelu = relay.multiply(gelu, const1)
    gelu = relay.add(gelu, bias)
    gelu = relay.multiply(gelu, const2)
    gelu = relay.tanh(gelu)
    gelu = relay.add(gelu, relay.const(1, dtype="float32"))
    gelu = relay.multiply(gelu, relay.const(0.5))
    res = relay.multiply(gelu, bias)

    return relay.Function([x, y, z, b], res)

def matmul_bias_mul_example():
    x = relay.var("x", relay.TensorType((4, 6), "float32"))
    y = relay.var("y", relay.TensorType((6, 3), "float32"))
    z = relay.var("z", relay.TensorType((3, 2), "float32"))
    b = relay.var("b", relay.TensorType((4, 2), "float32"))
    data_mul = relay.var("data_mul", relay.TensorType((4, 2), "float32"))
    matmul0 = nn.matmul(x, y)
    matmul1 = nn.matmul(matmul0, z)
    bias = relay.add(matmul1, b)
    mul = relay.multiply(bias, data_mul)

    return relay.Function([x, y, z, b, data_mul], mul)

def matmul_bias_mul_add_example():
    x = relay.var("x", relay.TensorType((4, 6), "float32"))
    y = relay.var("y", relay.TensorType((6, 3), "float32"))
    z = relay.var("z", relay.TensorType((3, 2), "float32"))
    b = relay.var("b", relay.TensorType((4, 2), "float32"))
    data_mul = relay.var("data_mul", relay.TensorType((4, 2), "float32"))
    data_add = relay.var("data_add", relay.TensorType((4, 2), "float32"))
    matmul0 = nn.matmul(x, y)
    matmul1 = nn.matmul(matmul0, z)
    bias = relay.add(matmul1, b)
    mul = relay.multiply(bias, data_mul)
    add = relay.add(mul, data_add)

    return relay.Function([x, y, z, b, data_mul, data_add], add)

def benchmark(batch_size=1, batches=10, warmup=2):
    ctx = tvm.cpu()

    f = matmul_bias_mul_add_example()
    mod = tvm.IRModule.from_expr(f)
    print(mod['main'].astext(show_meta_data=False))

    mod = relay.transform.MergeComposite(pattern_table())(mod)
    print(mod['main'].astext(show_meta_data=False))
    mod = relay.transform.AnnotateTarget(["dnnl"])(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    # print(mod['main'].astext(show_meta_data=False))

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

    datax = np.random.uniform(size=(4, 6)) - 0.5
    datay = np.random.uniform(size=(6, 3)) - 0.5
    dataz = np.random.uniform(size=(3, 2)) - 0.5
    datab = np.random.uniform(size=(4, 2)) - 0.5
    datamul = np.random.uniform(size=(4, 2)) - 0.5
    dataadd = np.random.uniform(size=(4, 2)) - 0.5

    # print(datax)
    # print(datay)
    # print(dataz)
    # print(datab)
    # print(datamul)

    rt_mod.set_input("x", tvm.nd.array(datax.astype("float32")))
    rt_mod.set_input("y", tvm.nd.array(datay.astype("float32")))
    rt_mod.set_input("z", tvm.nd.array(dataz.astype("float32")))
    rt_mod.set_input("b", tvm.nd.array(datab.astype("float32")))
    rt_mod.set_input("data_mul", tvm.nd.array(datamul.astype("float32")))
    rt_mod.set_input("data_add", tvm.nd.array(dataadd.astype("float32")))
    rt_mod.get_output(0).copyfrom(tvm.nd.array(dataadd.astype("float32")))
    rt_mod.run()
    tvm_output = rt_mod.get_output(0)
    print(tvm_output)
    # numpy relu results
    # print(np.maximum(np.matmul(np.matmul(datax, datay), dataz) + datab, 0))
    # print((np.matmul(np.matmul(datax, datay), dataz) + datab) * datamul)
    print((np.matmul(np.matmul(datax, datay), dataz) + datab) * datamul + dataadd)

benchmark(batch_size=1)