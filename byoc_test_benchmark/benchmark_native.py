import os

import mxnet as mx
import tvm

import gluonnlp
import numpy as np

from tvm.relay.op.contrib.dnnl import *
from tvm import relay, auto_scheduler
import tvm.contrib.graph_executor as runtime
from tvm.relay.testing import *

batch_size = 1
seq_length = 128

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

target = "llvm -mcpu=cascadelake -libs=mkl"

# logdir = "/home/mengceng/workspace/TLCBench/tmp_logs"
# log_file = os.path.join(
#     logdir, "autoscheduler", "unknown", "bert-B1-float32" + ".json")

ctx = tvm.cpu(0)

# with auto_scheduler.ApplyHistoryBest(log_file):
#     with tvm.transform.PassContext(opt_level=3,
#      config={"relay.backend.use_auto_scheduler": True}):
#         lib = relay.build(mod, target=target, params=params)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

module = runtime.GraphModule(lib["default"](ctx))

# Feed input data
data = np.random.uniform(size=input_shape[0])
token_types = np.random.uniform(size=input_shape[1])
valid_length = np.array([seq_length] * batch_size)
module.set_input(data0=data, data1=token_types, data2=valid_length)

# test accuracy
# module.run()
# tvm_output_0 = module.get_output(0).numpy()
# tvm_output_1 = module.get_output(1).numpy()
# seq_encoding, cls_encoding  = model(mx.nd.array(data), mx.nd.array(token_types), mx.nd.array(valid_length))
# np.testing.assert_allclose(seq_encoding.asnumpy(), tvm_output_0, rtol=1e-04, atol=1e-04)
# np.testing.assert_allclose(cls_encoding.asnumpy(), tvm_output_1, rtol=1e-04, atol=1e-04)
# print("passed")

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

