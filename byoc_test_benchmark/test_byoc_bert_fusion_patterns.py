import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay
from tvm.relay.testing import *
from tvm.relay import nn

from tvm.topi.utils import get_const_tuple
from tvm.relay.testing.temp_op_attr import TempOpAttr

import numpy as np
import math

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
    x = relay.var("x", relay.TensorType((128, 2, 768), "float32"))
    y = relay.var("y", relay.TensorType((2304, 768), "float32"))
    matmul0 = nn.special_matmul(x, y)

    return relay.Function([x, y], matmul0)

def dense_bias_example():
    x = relay.var("x", relay.TensorType((128, 2, 768), "float32"))
    y = relay.var("y", relay.TensorType((2304, 768), "float32"))
    b = relay.var("b", relay.TensorType((2304,), "float32"))
    matmul0 = nn.special_matmul(x, y)
    bias = relay.add(matmul0, b)

    return relay.Function([x, y, b], bias)

def dense_bias_gelu_example():
    x = relay.var("x", relay.TensorType((128, 2, 768), "float32"))
    y = relay.var("y", relay.TensorType((2304, 768), "float32"))
    b = relay.var("b", relay.TensorType((2304,), "float32"))
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

def check_correctness(func):
    ctx = tvm.cpu()
    f = func()
    mod = tvm.IRModule.from_expr(f)

    # mod = relay.transform.CanonicalizeOps()(mod)
    mod = relay.transform.InferType()(mod)
    # mod = relay.transform.SimplifyInference()(mod)
    # mod = relay.transform.FoldConstant()(mod)
    # mod = relay.transform.FoldScaleAxis()(mod)
    mod = relay.transform.FoldConstant()(mod)
    with TempOpAttr("nn.special_matmul", "FTVMAlterOpLayout", alter_special_matmul):
        mod = relay.transform.AlterOpLayout()(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.MergeComposite(pattern_table())(mod)
    mod = relay.transform.AnnotateTarget(["dnnl"])(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)

    json, lib, params = relay.build(mod, "llvm")
    rt_mod = tvm.contrib.graph_executor.create(json, lib, ctx)

    datax = np.random.uniform(size=(128, 2, 768)) - 0.5
    datay = np.random.uniform(size=(768, 2304)) - 0.5
    if func != dense_example:
        datab = np.random.uniform(size=(2304)) - 0.5

    rt_mod.set_input("x", tvm.nd.array(datax.astype("float32")))
    rt_mod.set_input("y", tvm.nd.array(datay.transpose().astype("float32")))
    if func != dense_example:
        rt_mod.set_input("b", tvm.nd.array(datab.astype("float32")))
    rt_mod.run()
    tvm_output = rt_mod.get_output(0).numpy()

    if func == dense_example:
        ans = np.matmul(datax.reshape((128 * 2, 768)), datay).reshape(128, 2, 2304)
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_example: passed\n")
    if func == dense_bias_example:
        ans = (np.matmul(datax.reshape((128 * 2, 768)), datay) + datab).reshape(128, 2, 2304)
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_bias_example: passed\n")
    if func == dense_bias_gelu_example:
        ans0 = np.matmul(datax.reshape((128 * 2, 768)), datay) + datab
        ans1 = ans0 / 1.41421
        ans1 = np.array([math.erf(x) for x in ans1.flatten().tolist()]).reshape(ans1.shape)
        ans2 = 0.5 * ans0
        ans1 = ans1 + 1.0
        ans = ans1 * ans2
        ans = ans.reshape((128, 2, 2304))
        np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
        print("dense_bias_gelu_example: passed\n")

check_correctness(dense_example)
check_correctness(dense_bias_example)
check_correctness(dense_bias_gelu_example)