'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import time
import mxnet as mx
import warnings

import math

from tvm.relay.op.tensor import bitwise_and

from tvm.relay.testing.temp_op_attr import TempOpAttr

import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay
import tvm.contrib.graph_executor as runtime
import numpy as np
from tvm.relay.testing import *
import os
from tvm.contrib import utils
from tvm.relay import nn

from tvm.topi.utils import get_const_tuple

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

weight_dic = {"A":"N",
              "B":"C",
              "C":"H",
              "D":"W",
              "a":"n",
              "b":"c",
              "c":"h",
              "d":"w"}

@relay.op.register_alter_op_layout("nn.special_matmul", level=114)
def alter_special_matmul(attrs, inputs, tinfos, out_type):
    new_attrs = dict(attrs)

    data, weight = inputs
    data_tensor, weight_tensor = tinfos

    if new_attrs['is_batch_matmul']:
        B1, B2, M, K = get_const_tuple(data_tensor.shape)
        _, _, N, _ = get_const_tuple(weight_tensor.shape)
        res = relay.query_layout.AutoQuery_batch_matmul(B1, B2, M, K, N)
    else:
        B, M, IC = get_const_tuple(data_tensor.shape)
        OC, IC = get_const_tuple(weight_tensor.shape)
        B = B * M
        res = relay.query_layout.AutoQuery_innerproduct(B, IC, OC)

    print("queried weight layout:", res)

    _, weight_df, _, _ = res.split(',')

    def trans_data(input_data, is_weight=False):
        dic = weight_dic
        input_str = [c for c in input_data]
        output_list = []
        
        for c in input_str:
            if c in dic.keys():
                output_list += dic[c]
            else:
                output_list += c

        all_lower_case = True
        for c in output_list:
            if c == c.upper():
                all_lower_case = False
                break
        
        if all_lower_case:
            for i in range(len(output_list)):
                output_list[i] = output_list[i].upper()

        output_str = ""
        for c in output_list:
            output_str += c

        return output_str

    print("translated weight layout:", trans_data(weight_df, is_weight=True))
    new_attrs['weight_layout'] = trans_data(weight_df, is_weight=True)

    return relay.nn.special_matmul(data, weight, **new_attrs)

def dense_example():
    x = relay.var("x", relay.TensorType((2, 14, 768), "float32"))
    y = relay.var("y", relay.TensorType((16, 768), "float32"))
    matmul0 = nn.special_matmul(x, y)

    return relay.Function([x, y], matmul0)

def dense_bias_example():
    x = relay.var("x", relay.TensorType((2, 14, 768), "float32"))
    y = relay.var("y", relay.TensorType((16, 768), "float32"))
    b = relay.var("b", relay.TensorType((16,), "float32"))
    matmul0 = nn.special_matmul(x, y)
    bias = relay.add(matmul0, b)

    return relay.Function([x, y, b], bias)

def dense_bias_relu_example():
    x = relay.var("x", relay.TensorType((2, 14, 768), "float32"))
    y = relay.var("y", relay.TensorType((16, 768), "float32"))
    b = relay.var("b", relay.TensorType((16,), "float32"))
    matmul0 = nn.special_matmul(x, y)
    bias = relay.add(matmul0, b)
    # relu
    res = nn.relu(bias)

    return relay.Function([x, y, b], res)

def dense_bias_gelu_example():
    x = relay.var("x", relay.TensorType((2, 14, 768), "float32"))
    y = relay.var("y", relay.TensorType((16, 768), "float32"))
    b = relay.var("b", relay.TensorType((16,), "float32"))
    matmul0 = nn.special_matmul(x, y)
    bias = relay.add(matmul0, b)
    # gelu_erf
    const1 = relay.const(1.41421)
    const2 = relay.const(0.5)
    const3 = relay.const(1.0)
    div = relay.divide(bias, const1)
    erf = relay.erf(div)
    mul = relay.multiply(bias, const2)
    add = relay.add(erf, const3)
    mul2 = relay.multiply(mul, add)

    return relay.Function([x, y, b], mul2)

def dense_bias_mul_example():
    x = relay.var("x", relay.TensorType((2, 14, 768), "float32"))
    y = relay.var("y", relay.TensorType((16, 768), "float32"))
    b = relay.var("b", relay.TensorType((16,), "float32"))
    data_mul = relay.var("data_mul", relay.TensorType((2, 14, 16), "float32"))
    matmul0 = nn.special_matmul(x, y)
    bias = relay.add(matmul0, b)
    mul = relay.multiply(bias, data_mul)

    return relay.Function([x, y, b, data_mul], mul)

def dense_bias_mul_add_example():
    x = relay.var("x", relay.TensorType((2, 14, 768), "float32"))
    y = relay.var("y", relay.TensorType((16, 768), "float32"))
    b = relay.var("b", relay.TensorType((16,), "float32"))
    data_mul = relay.var("data_mul", relay.TensorType((2, 14, 16), "float32"))
    data_add = relay.var("data_add", relay.TensorType((2, 14, 16), "float32"))
    matmul0 = nn.special_matmul(x, y)
    bias = relay.add(matmul0, b)
    mul = relay.multiply(bias, data_mul)
    add = relay.add(mul, data_add)

    return relay.Function([x, y, b, data_mul, data_add], add)

def check_correctness(func):
    ctx = tvm.cpu()
    f = func()
    mod = tvm.IRModule.from_expr(f)
    # print(mod['main'].astext(show_meta_data=False))

    mod = relay.transform.CanonicalizeOps()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.SimplifyInference()(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.FoldScaleAxis()(mod)
    mod = relay.transform.FoldConstant()(mod)
    with TempOpAttr("nn.special_matmul", "FTVMAlterOpLayout", alter_special_matmul):
        mod = relay.transform.AlterOpLayout()(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.MergeComposite(pattern_table())(mod)
    mod = relay.transform.AnnotateTarget(["dnnl"])(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    # print(mod)
    # print(mod['main'].astext(show_meta_data=False))

    json, lib, params = relay.build(mod, "llvm")
    rt_mod = tvm.contrib.graph_executor.create(json, lib, ctx)#Create a runtime executor module given a graph and module.

    datax = np.random.uniform(size=(2, 14, 768)) - 0.5
    datay = np.random.uniform(size=(768, 16)) - 0.5
    if func != dense_example:
        datab = np.random.uniform(size=(16,)) - 0.5
    if func == dense_bias_mul_example or func == dense_bias_mul_add_example:
        datamul = np.random.uniform(size=(2, 14, 16)) - 0.5
    if func == dense_bias_mul_add_example:
        dataadd = np.random.uniform(size=(2, 14, 16)) - 0.5

    rt_mod.set_input("x", tvm.nd.array(datax.astype("float32")))
    rt_mod.set_input("y", tvm.nd.array(datay.transpose().astype("float32")))
    if func != dense_example:
        rt_mod.set_input("b", tvm.nd.array(datab.astype("float32")))
    if func == dense_bias_mul_example or func == dense_bias_mul_add_example:
        rt_mod.set_input("data_mul", tvm.nd.array(datamul.astype("float32")))
    if func == dense_bias_mul_add_example:
        rt_mod.set_input("data_add", tvm.nd.array(dataadd.astype("float32")))
    if func == dense_bias_mul_example:
        rt_mod.get_output(0).copyfrom(tvm.nd.array(datamul.astype("float32")))
    if func == dense_bias_mul_add_example:
        rt_mod.get_output(0).copyfrom(tvm.nd.array(dataadd.astype("float32")))
    rt_mod.run()
    tvm_output = rt_mod.get_output(0).numpy()

    if func == dense_example:
        ans = np.matmul(datax.reshape((2 * 14, 768)), datay).reshape(2, 14, 16)
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_example: passed\n")
    if func == dense_bias_example:
        ans = (np.matmul(datax.reshape((2 * 14, 768)), datay) + datab).reshape(2, 14, 16)
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_bias_example: passed\n")
    if func == dense_bias_relu_example:
        ans = np.maximum(np.matmul(datax.reshape((2 * 14, 768)), datay) + datab, 0).reshape(2, 14, 16)
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_bias_relu_example: passed\n")
    if func == dense_bias_gelu_example:
        ans0 = np.matmul(datax.reshape((2 * 14, 768)), datay) + datab
        ans1 = ans0 / 1.41421
        ans1 = np.array([math.erf(x) for x in ans1.flatten().tolist()]).reshape(ans1.shape)
        ans2 = 0.5 * ans0
        ans1 = ans1 + 1.0
        ans = ans1 * ans2
        ans = ans.reshape((2, 14, 16))
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_bias_gelu_example: passed\n")
    if func == dense_bias_mul_example:
        ans = ((np.matmul(datax.reshape((2 * 14, 768)), datay) + datab) * datamul.reshape(2 * 14, 16)).reshape(2, 14, 16)
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_bias_mul_example: passed\n")
    if func == dense_bias_mul_add_example:
        ans = ((np.matmul(datax.reshape((2 * 14, 768)), datay) + datab) * datamul.reshape(2 * 14, 16) + dataadd.reshape(2 * 14, 16)).reshape(2, 14, 16)
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_bias_mul_add_example: passed\n")

def dense_batch_matmul_example():
    x = relay.var("x", relay.TensorType((1, 12, 14, 14), "float32"))
    y = relay.var("y", relay.TensorType((1, 12, 14, 64), "float32"))
    matmul0 = nn.special_matmul(x, y, "NCHW", is_batch_matmul=True)

    return relay.Function([x, y], matmul0)

def dense_batch_matmul_div_add_example():
    x = relay.var("x", relay.TensorType((1, 12, 14, 14), "float32"))
    y = relay.var("y", relay.TensorType((1, 12, 14, 64), "float32"))
    data_div = relay.var("data_div", relay.TensorType((1, 12, 14, 64), "float32"))
    data_add = relay.var("data_add", relay.TensorType((1, 12, 14, 64), "float32"))
    matmul0 = nn.special_matmul(x, y, "NCHW", is_batch_matmul=True)
    div = relay.divide(matmul0, data_div)
    add = relay.add(div, data_add)

    return relay.Function([x, y, data_div, data_add], add)

def check_batch_matmul_correctness(func):
    ctx = tvm.cpu()
    f = func()
    mod = tvm.IRModule.from_expr(f)
    # print(mod['main'].astext(show_meta_data=False))

    # print(mod)
    mod = relay.transform.CanonicalizeOps()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.SimplifyInference()(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.FoldScaleAxis()(mod)
    mod = relay.transform.FoldConstant()(mod)
    with TempOpAttr("nn.special_matmul", "FTVMAlterOpLayout", alter_special_matmul):
        mod = relay.transform.AlterOpLayout()(mod)
    # print(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.MergeComposite(pattern_table())(mod)
    mod = relay.transform.AnnotateTarget(["dnnl"])(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    # print(mod)
    # print(mod['main'].astext(show_meta_data=False))

    json, lib, params = relay.build(mod, "llvm")
    rt_mod = tvm.contrib.graph_executor.create(json, lib, ctx)#Create a runtime executor module given a graph and module.

    datax = np.random.uniform(size=(1, 12, 14, 14)) - 0.5
    datay = np.random.uniform(size=(1, 12, 14, 64)) - 0.5
    data_div = np.random.uniform(size=(1, 12, 14, 64)) - 0.5
    data_add = np.random.uniform(size=(1, 12, 14, 64)) - 0.5

    rt_mod.set_input("x", tvm.nd.array(datax.astype("float32")))
    rt_mod.set_input("y", tvm.nd.array(datay.astype("float32")))
    if func == dense_batch_matmul_div_add_example:
        rt_mod.set_input("data_div", tvm.nd.array(data_div.astype("float32")))
        rt_mod.set_input("data_add", tvm.nd.array(data_add.astype("float32")))
    rt_mod.run()
    tvm_output = rt_mod.get_output(0).numpy()

    ans = np.matmul(datax, datay)
    if func == dense_batch_matmul_div_add_example:
        ans = (ans / data_div) + data_add
    np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
    print("batch_matmul_example: passed\n")

check_correctness(dense_example)
check_correctness(dense_bias_example)
check_correctness(dense_bias_relu_example)
check_correctness(dense_bias_gelu_example)
check_correctness(dense_bias_mul_example)
check_correctness(dense_bias_mul_add_example)
check_batch_matmul_correctness(dense_batch_matmul_example)
check_batch_matmul_correctness(dense_batch_matmul_div_add_example)
