import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay
from tvm.relay.testing import *
from tvm.relay import nn

from tvm.topi.utils import get_const_tuple
from tvm.relay.testing.temp_op_attr import TempOpAttr

import tvm.contrib.graph_executor as runtime

import numpy as np

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
        print(data_tensor.shape)
        print(weight_tensor.shape)
        B, M, K = get_const_tuple(data_tensor.shape)
        _, _, N = get_const_tuple(weight_tensor.shape)
        res = relay.query_layout.AutoQuery_batch_matmul(B, M, K, N)
    else:
        B, M, IC = get_const_tuple(data_tensor.shape)
        OC, IC = get_const_tuple(weight_tensor.shape)
        B = B * M
        res = relay.query_layout.AutoQuery_innerproduct(B, IC, OC)

    print("queried weight layout:", res)

    if new_attrs['is_batch_matmul']:
        _, weight_df, _ = res.split(',')
    else:
        _, weight_df, _, _ = res.split(',')

    def trans_data(input_data):
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
    print("translated weight layout:", trans_data(weight_df))
    new_attrs['weight_layout'] = trans_data(weight_df)

    return relay.nn.special_matmul(data, weight, **new_attrs)


@relay.op.register_alter_op_layout("nn.batch_matmul", level=114)
def alter_batch_matmul(attrs, inputs, tinfos, out_type):
    new_attrs = dict(attrs)

    data, weight = inputs
    data_tensor, weight_tensor = tinfos

    print(data_tensor.shape)
    print(weight_tensor.shape)
    B, M, K = get_const_tuple(data_tensor.shape)
    _, N, _ = get_const_tuple(weight_tensor.shape)
    res = relay.query_layout.AutoQuery_batch_matmul(B, M, K, N)

    print("queried weight layout:", res)

    _, weight_df, _ = res.split(',')

    def trans_data(input_data):
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
    print("translated weight layout:", trans_data(weight_df))
    new_attrs['weight_layout'] = trans_data(weight_df)

    return relay.nn.batch_matmul(data, weight, **new_attrs)

mb=128

ic1=768
oc1=768

ic2=768
oc2=3072

ic3=3072
oc3=768

ic4=768
oc4=2304

def dense_example():
    x1 = relay.var("x1", relay.TensorType((mb, 1, ic1), "float32"))
    y1 = relay.var("y1", relay.TensorType((oc1, ic1), "float32"))
    x2 = relay.var("x2", relay.TensorType((mb, 1, ic2), "float32"))
    y2 = relay.var("y2", relay.TensorType((oc2, ic2), "float32"))
    x3 = relay.var("x3", relay.TensorType((mb, 1, ic3), "float32"))
    y3 = relay.var("y3", relay.TensorType((oc3, ic3), "float32"))
    x4 = relay.var("x4", relay.TensorType((mb, 1, ic4), "float32"))
    y4 = relay.var("y4", relay.TensorType((oc4, ic4), "float32"))
    matmul0 = nn.special_matmul(x1, y1)
    matmul1 = nn.special_matmul(x2, y2)
    matmul2 = nn.special_matmul(x3, y3)
    matmul3 = nn.special_matmul(x4, y4)
    res = relay.Tuple([matmul0, matmul1, matmul2, matmul3])

    return relay.Function([x1, y1, x2, y2, x3, y3, x4, y4], res)

def dense_bias_example():
    x1 = relay.var("x1", relay.TensorType((mb, 1, ic1), "float32"))
    y1 = relay.var("y1", relay.TensorType((oc1, ic1), "float32"))
    b1 = relay.var("b1", relay.TensorType((oc1,), "float32"))
    x2 = relay.var("x2", relay.TensorType((mb, 1, ic2), "float32"))
    y2 = relay.var("y2", relay.TensorType((oc2, ic2), "float32"))
    b2 = relay.var("b2", relay.TensorType((oc2,), "float32"))
    x3 = relay.var("x3", relay.TensorType((mb, 1, ic3), "float32"))
    y3 = relay.var("y3", relay.TensorType((oc3, ic3), "float32"))
    b3 = relay.var("b3", relay.TensorType((oc3,), "float32"))
    x4 = relay.var("x4", relay.TensorType((mb, 1, ic4), "float32"))
    y4 = relay.var("y4", relay.TensorType((oc4, ic4), "float32"))
    b4 = relay.var("b4", relay.TensorType((oc4,), "float32"))
    matmul0 = nn.special_matmul(x1, y1)
    bias0 = relay.add(matmul0, b1)
    matmul1 = nn.special_matmul(x2, y2)
    bias1 = relay.add(matmul1, b2)
    matmul2 = nn.special_matmul(x3, y3)
    bias2 = relay.add(matmul2, b3)
    matmul3 = nn.special_matmul(x4, y4)
    bias3 = relay.add(matmul3, b4)
    res = relay.Tuple([bias0, bias1, bias2, bias3])

    return relay.Function([x1, y1, b1, x2, y2, b2, x3, y3, b3, x4, y4, b4], res)

def perf(func):
    ctx = tvm.cpu()
    f = func()
    mod = tvm.IRModule.from_expr(f)

    datax1 = np.random.uniform(size=(mb, 1, ic1)) - 0.5
    datay1 = np.random.uniform(size=(ic1, oc1)) - 0.5
    datax2 = np.random.uniform(size=(mb, 1, ic2)) - 0.5
    datay2 = np.random.uniform(size=(ic2, oc2)) - 0.5
    datax3 = np.random.uniform(size=(mb, 1, ic3)) - 0.5
    datay3 = np.random.uniform(size=(ic3, oc3)) - 0.5
    datax4 = np.random.uniform(size=(mb, 1, ic4)) - 0.5
    datay4 = np.random.uniform(size=(ic4, oc4)) - 0.5
    if func != dense_example:
        datab1 = np.random.uniform(size=(oc1)) - 0.5
        datab2 = np.random.uniform(size=(oc2)) - 0.5
        datab3 = np.random.uniform(size=(oc3)) - 0.5
        datab4 = np.random.uniform(size=(oc4)) - 0.5

    params = {}
    params["y1"] = tvm.nd.array(datay1.transpose().astype("float32"))
    params["y2"] = tvm.nd.array(datay2.transpose().astype("float32"))
    params["y3"] = tvm.nd.array(datay3.transpose().astype("float32"))
    params["y4"] = tvm.nd.array(datay4.transpose().astype("float32"))

    BindPass = tvm.relay.transform.function_pass(
        lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
            fn, params
        ),
        opt_level=1,
    )
    mod = BindPass(mod)
    mod = relay.transform.FoldConstant()(mod)
    with TempOpAttr("nn.special_matmul", "FTVMAlterOpLayout", alter_special_matmul):
        mod = relay.transform.AlterOpLayout()(mod)
    mod = relay.transform.FoldConstant()(mod)

    mod = relay.transform.MergeComposite(pattern_table())(mod)
    mod = relay.transform.AnnotateTarget(["dnnl"])(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    print(mod)

    target = "llvm -mcpu=cascadelake"
    lib = relay.build(mod, target=target)
    rt_mod = runtime.GraphModule(lib["default"](ctx))

    rt_mod.set_input("x1", tvm.nd.array(datax1.astype("float32")))
    # rt_mod.set_input("y1", tvm.nd.array(datay1.transpose().astype("float32")))
    rt_mod.set_input("x2", tvm.nd.array(datax2.astype("float32")))
    # rt_mod.set_input("y2", tvm.nd.array(datay2.transpose().astype("float32")))
    rt_mod.set_input("x3", tvm.nd.array(datax3.astype("float32")))
    # rt_mod.set_input("y3", tvm.nd.array(datay3.transpose().astype("float32")))
    rt_mod.set_input("x4", tvm.nd.array(datax4.astype("float32")))
    # rt_mod.set_input("y4", tvm.nd.array(datay4.transpose().astype("float32")))
    if func != dense_example:
        rt_mod.set_input("b1", tvm.nd.array(datab1.astype("float32")))
        rt_mod.set_input("b2", tvm.nd.array(datab2.astype("float32")))
        rt_mod.set_input("b3", tvm.nd.array(datab3.astype("float32")))
        rt_mod.set_input("b4", tvm.nd.array(datab4.astype("float32")))
    
    import time
    def warmup():
        for i in range(200):
            rt_mod.run()
        ctx.sync()

    def x():
        for i in range(1000):
            if i == 999:
                print("start here")
            rt_mod.run()
        ctx.sync()

    warmup()
    start = time.time()
    x()
    end = time.time()
    print("time:", (end-start)/1000)

    # tvm_output = rt_mod.get_output(0).numpy()

    # if func == dense_example:
    #     ans = np.matmul(datax.reshape((128 * 2, 768)), datay).reshape(128, 2, 2304)
    #     np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
    #     print("dense_example: passed\n")
    # if func == dense_bias_example:
    #     ans = (np.matmul(datax.reshape((128 * 2, 768)), datay) + datab).reshape(128, 2, 2304)
    #     np.testing.assert_allclose(ans, tvm_output, rtol=1e-05, atol=1e-05)
    #     print("dense_bias_example: passed\n")

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

# perf(dense_example)
perf(dense_bias_example)