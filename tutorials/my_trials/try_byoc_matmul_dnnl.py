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

# model_dict = {'resnet50_v1': resnet50_v1}#{'mobilenet_v2_1_0': mobilenet_v2_1_0}
# model_dict = {'resnet50_v1': resnet}
# model_dict = {'resnet50_v1': resnet, 'mobilenet_v2_1_0': mobilenet}

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

def make_pattern(with_bias=True, with_bn=False):
    from tvm.relay.dataflow_pattern import is_op, wildcard
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    gamma, beta, moving_mean, moving_var = wildcard(), wildcard(), wildcard(), wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("nn.bias_add")(conv, bias)
    else:
        conv_out = conv
    if with_bn:
        bn_out = is_op("nn.batch_norm")(conv_out, gamma, beta, moving_mean, moving_var)
    else:
        bn_out = conv_out
    return is_op("nn.relu")(bn_out)

# def make_pattern(with_bias=True):
#     from tvm.relay.dataflow_pattern import is_op, wildcard
#     data = wildcard()
#     weight = wildcard()
#     conv = is_op('nn.conv2d')(data, weight)
#     return wildcard()(conv)

# conv2d_bias_relu_pat = ("dnnl.conv2d_relu_with_bias", make_pattern(with_bias=True))
# conv2d_bias_bn_relu_pat = ("dnnl.conv2d_bn_relu_with_bias", make_pattern(with_bias=True, with_bn=True))
# conv2d_relu_pat = ("dnnl.conv2d_relu_wo_bias", make_pattern(with_bias=False))
# conv2d_bn_relu_pat = ("dnnl.conv2d_bn_relu_wo_bias", make_pattern(with_bias=False, with_bn=True))
# patterns = [conv2d_bias_relu_pat, conv2d_relu_pat, conv2d_bias_bn_relu_pat, conv2d_bn_relu_pat]#



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

# class Model(HybridBlock):
#     def __init__(self, **kwargs):
#         super(Model, self).__init__(**kwargs)
#         # use name_scope to give child Blocks appropriate names.
#         # with self.name_scope():
#         self.relu = nn.Activation('relu')

#     def hybrid_forward(self, x, y, z):
#         matmul0 = nn.matmul(x, y)
#         matmul1 = nn.matmul(matmul0, z)
#         res = self.relu(matmul1)
#         return res

def example():
    x = relay.var("x", relay.TensorType((2, 3), "float32"))
    y = relay.var("y", relay.TensorType((3, 4), "float32"))
    z = relay.var("z", relay.TensorType((4, 1), "float32"))
    shape = (2, 4)
    matmul0 = nn.matmul(x, y)
    matmul1 = nn.matmul(matmul0, z)
    # res = nn.relu(matmul1)
    return relay.Function([x, y, z], matmul1)

def benchmark(batch_size=1, batches=10, warmup=2):
    ctx = tvm.cpu()

    f = example()
    mod = tvm.IRModule.from_expr(f)
    # print(mod['main'].astext(show_meta_data=False))

    seq = tvm.transform.Sequential(
        [
            #relay.transform.RemoveUnusedFunctions(),
            #tvm.transform.PrintIR(),
            #relay.transform.AlterOpLayout(),
            #tvm.transform.PrintIR(),
            #relay.transform.FoldConstant(),
            #tvm.transform.PrintIR(),
            #relay.transform.AnnotateTarget("dnnl"),
            #tvm.transform.PrintIR(),
            #relay.transform.MergeCompilerRegions(),
            #tvm.transform.PrintIR(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        with tvm.target.Target("llvm"):
            mod = seq(mod)
    # print(mod['main'].astext(show_meta_data=False))

    """
    seq = tvm.transform.Sequential(
        [
            # transform.InferType(),
            transform.RemoveUnusedFunctions(),
            relay.transform.AlterOpLayout(),
            # transform.ConvertLayout(
            #     {
            #         "nn.conv2d": ["NCHW", "OIHW"],
            #     #     "nn.conv3d": ["NCDHW", "default"],
            #     #     "nn.conv2d_transpose": ["NCHW", "default"],
            #     }
            # ),
            # transform.FoldConstant(),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            # transform.InferType(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):#
        with tvm.target.Target("llvm"):
            mod = seq(mod)
    """

    # with relay.build_config(opt_level=3):
    #     graph, lib, params = relay.build(mod, target, params=params)

    json, lib, params = relay.build(mod, "llvm")
    rt_mod = tvm.contrib.graph_executor.create(json, lib, ctx)#Create a runtime executor module given a graph and module.

    datax = np.random.uniform(size=(2, 3))
    datay = np.random.uniform(size=(3, 4))
    dataz = np.random.uniform(size=(4, 1))
    # datax = np.ones(shape=(2, 3))
    # datay = np.ones(shape=(3, 4))
    # dataz = np.ones(shape=(4, 1))

    rt_mod.set_input("x", tvm.nd.array(datax.astype("float32")))
    rt_mod.set_input("y", tvm.nd.array(datay.astype("float32")))
    rt_mod.set_input("z", tvm.nd.array(dataz.astype("float32")))
    rt_mod.run()
    tvm_output = rt_mod.get_output(0)
    print(tvm_output)

benchmark(batch_size=1)  

# def benchmark(batch_size=1, batches=10, warmup=2):
#     mx.random.seed(0)
#     sample = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size,3,224,224))
#     target = "llvm -model=platinum-8124m -mcpu=skylake-avx512"
#     ctx = tvm.cpu()

#     input_shape = (batch_size, 3, 224, 224)
#     for model_name in model_dict.keys():
#         # net = model_dict[model_name](pretrained=True)
#         # net.hybridize(static_alloc=True, static_shape=True)
#         # mod, params = relay.frontend.from_mxnet(net, shape={"data": input_shape}, dtype="float32")#port the Gluon model to a portable computational graph
#         mod, params = model_dict[model_name].get_workload(batch_size=batch_size, dtype="float32")
#         # mod =relay.transform.AlterOpLayout()(mod)
#         # print('==================1 relayed model ==================')
#         # print(mod["main"].astext(show_meta_data=False))
#         # mod2 = relay.transform.MergeComposite(pattern_table())(mod)
#         # # print('==================2 MergeComposite ==================')
#         # # print(mod2["main"].astext(show_meta_data=False))
#         # mod3 = relay.transform.AnnotateTarget(["dnnl"])(mod2)
#         # print('==================3 AnnotateTarget ==================')
#         # # print(mod3["main"].astext(show_meta_data=False))
#         # # print(mod3)

#         # mod4 = relay.transform.MergeCompilerRegions()(mod3)
#         # print('==================4 MergeCompilerRegions ==================')
#         # # print(mod4["main"].astext(show_meta_data=False))
#         # # print(mod4)

#         # mod5 = relay.transform.PartitionGraph()(mod4)
#         # print('==================5 PartitionGraph ==================')
#         # # print(mod5["main"].astext(show_meta_data=False))
#         # # print(mod5)

#         # mod6 =relay.transform.AlterOpLayout()(mod5)

#     #     seq = tvm.transform.Sequential(
#     #     [
#     #         # transform.InferType(),
#     #         relay.transform.MergeComposite(pattern_table()),
#     #         relay.transform.AnnotateTarget(["dnnl"]),
#     #         relay.transform.MergeCompilerRegions(),
#     #         relay.transform.PartitionGraph(),
#     #         relay.transform.RemoveUnusedFunctions(),
#     #         relay.transform.AlterOpLayout()
#     #         # relay.transform.ConvertLayout(
#     #         #     {
#     #         #         "nn.conv2d": ["NCHW", "OIHW"],
#     #         #     }
#     #         # ),
#     #         # relay.transform.FoldConstant(),
#     #     ]
#     # )

#         seq = tvm.transform.Sequential(
#             [
#                 # transform.InferType(),
#                 relay.transform.RemoveUnusedFunctions(),
#                 relay.transform.AlterOpLayout(),
#                 # transform.ConvertLayout(
#                 #     {
#                 #         "nn.conv2d": ["NCHW", "OIHW"],
#                 #     #     "nn.conv3d": ["NCDHW", "default"],
#                 #     #     "nn.conv2d_transpose": ["NCHW", "default"],
#                 #     }
#                 # ),
#                 # transform.FoldConstant(),
#                 relay.transform.AnnotateTarget("dnnl"),
#                 relay.transform.MergeCompilerRegions(),
#                 relay.transform.PartitionGraph(),
#                 # transform.InferType(),
#             ]
#         )

#         with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):#
#             with tvm.target.Target("llvm"):
#                 mod = seq(mod)

#         # with tvm.transform.PassContext(opt_level=3):#compile the graph , instruments=[PrintIR()]
#         #     json, lib, param = tvm.relay.build(mod, target="llvm", params=params)
#         with relay.build_config(opt_level=3):
#             json, lib, params = relay.build(mod, "llvm", params=params)
#         # lib = update_lib(lib)
#         rt_mod = tvm.contrib.graph_executor.create(json, lib, ctx)#Create a runtime executor module given a graph and module.

#         data = np.random.uniform(size=input_shape)
#         # rt_mod.set_input("data", sample)
#         rt_mod.set_input("data", tvm.nd.array(data.astype("float32")))
#         for i in range(batches+warmup):
#             if i == warmup:
#                 tic = time.time()
#             out = rt_mod.run()
#         with_fuse_ms = (time.time() - tic) / (batches) * 1000
#         print("{}: with_fuse_ms: {:.4f} ms".format(model_name, with_fuse_ms))

# benchmark(batch_size=1) 