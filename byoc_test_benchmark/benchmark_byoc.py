import os

import mxnet as mx
import tvm

import gluonnlp
import numpy as np

from tvm.relay.op.contrib.dnnl import *
from tvm import relay
import tvm.contrib.graph_executor as runtime
from tvm.relay.testing import *

from tvm.contrib import utils
from tvm.relay import nn
from tvm.relay.expr import Call

from tvm.relay.testing.temp_op_attr import TempOpAttr

from tvm.topi.utils import get_const_tuple

batch_size = 1
seq_length = 128


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

    # print("queried weight layout:", res)

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

    # print("translated weight layout:", trans_data(weight_df, is_weight=True))
    new_attrs['weight_layout'] = trans_data(weight_df, is_weight=True)

    return relay.nn.special_matmul(data, weight, **new_attrs)


@relay.op.register_alter_op_layout("nn.batch_matmul", level=114)
def alter_batch_matmul(attrs, inputs, tinfos, out_type):
    new_attrs = dict(attrs)

    data, weight = inputs
    data_tensor, weight_tensor = tinfos

    B, M, K = get_const_tuple(data_tensor.shape)
    _, N, _ = get_const_tuple(weight_tensor.shape)
    res = relay.query_layout.AutoQuery_batch_matmul(B, M, K, N)
    # print("queried weight layout:", res)

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
    # print("translated weight layout:", trans_data(weight_df))
    new_attrs['weight_layout'] = trans_data(weight_df)

    return relay.nn.batch_matmul(data, weight, **new_attrs)

# Instantiate a BERT classifier using GluonNLP
model_name = "bert_12_768_12"
dataset = "book_corpus_wiki_en_uncased"
model, _ = gluonnlp.model.get_model(
    name=model_name,
    dataset_name=dataset,
    pretrained=True,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False,
)

# Convert the MXNet model into TVM Relay format
shape_dict = {
    "data0": (batch_size, seq_length),
    "data1": (batch_size, seq_length),
    "data2": (batch_size,),
}
mod, params = relay.frontend.from_mxnet(model, shape_dict)
input_shape = (shape_dict["data0"], shape_dict["data1"], shape_dict["data2"])

mod = tvm.relay.transform.FastMath()(mod)
mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
BindPass = tvm.relay.transform.function_pass(
    lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
        fn, params
    ),
    opt_level=1,
)
mod = BindPass(mod)
mod = tvm.relay.transform.FoldConstant()(mod)
mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
mod = tvm.relay.transform.FoldConstant()(mod)

with TempOpAttr("nn.special_matmul", "FTVMAlterOpLayout", alter_special_matmul):
    with TempOpAttr("nn.batch_matmul", "FTVMAlterOpLayout", alter_batch_matmul):
        mod = relay.transform.AlterOpLayout()(mod)
mod = relay.transform.FoldConstant()(mod)

mod = relay.transform.MergeComposite(pattern_table())(mod)
mod = relay.transform.AnnotateTarget(["dnnl"])(mod)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)
# print(mod)

target = "llvm -mcpu=cascadelake"

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

ctx = tvm.cpu(0)
module = runtime.GraphModule(lib["default"](ctx))

# Feed input data
data = np.random.uniform(size=input_shape[0])
token_types = np.random.uniform(size=input_shape[1])
valid_length = np.array([seq_length] * batch_size)
module.set_input(data0=data, data1=token_types, data2=valid_length)
# module.set_input(**params)
module.run()
tvm_output_0 = module.get_output(0).numpy()
tvm_output_1 = module.get_output(1).numpy()
seq_encoding, cls_encoding  = model(mx.nd.array(data), mx.nd.array(token_types), mx.nd.array(valid_length))
np.testing.assert_allclose(seq_encoding.asnumpy(), tvm_output_0, rtol=1e-04, atol=1e-04)
np.testing.assert_allclose(cls_encoding.asnumpy(), tvm_output_1, rtol=1e-04, atol=1e-04)
print("passed")
# print(tvm_output_0)
# print(seq_encoding.asnumpy())

# print(tvm_output_1)
# print(cls_encoding.asnumpy())

import time

def warmup():
    for i in range(200):
        module.run()
    ctx.sync()

def x():
    for i in range(1000):
        module.run()
    ctx.sync()

warmup()
start = time.time()
x()
end = time.time()
print("time:", (end-start)/1000)

