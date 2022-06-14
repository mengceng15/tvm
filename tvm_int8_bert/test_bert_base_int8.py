import os
import logging
import sys

from transformers import BertModel, BertTokenizer, BertConfig
import torch
import numpy as np

import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay, topi
import tvm.contrib.graph_executor as runtime
from tvm.relay.testing import *
from tvm.contrib import utils
from tvm.relay import nn

from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.relay.expr import Call
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.topi.utils import get_const_tuple


enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
text = ''.join([text] * 9) # 14 * 9 = 126
text += "pad pad" # 128
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 65
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0] * 64 + [1] * 64

batch_size = 1
if batch_size == 1:
    log_file = "int8_bert_base_latency.log"
else:
    assert batch_size == 128
    log_file = "int8_bert_base_throughput_bs128.log"

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens] * batch_size)
segments_tensors = torch.tensor([segments_ids] * batch_size)
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(vocab_size_or_config_json_file=30522, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

# # Instantiating the model
model = BertModel(config)

# # The model needs to be in evaluation mode
model.eval()

# # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
# # Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)
torch_result = traced_model(tokens_tensor, segments_tensors)

shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in list(traced_model.graph.inputs())[1:]]
mod_bert_fp32, params_bert_fp32 = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")

def quantize_relay_module(mod, params, qconfig=None, dataset=None):
	# default qconfig
	if not qconfig:
		qconfig = tvm.relay.quantize.qconfig()

	with qconfig:
		logging.debug('current quantize config')
		logging.debug(tvm.relay.quantize.current_qconfig())
		
		mod = tvm.relay.quantize.quantize(mod, params=params, dataset=dataset)

		logging.debug('after quantize')
	return mod

qconfig = tvm.relay.quantize.qconfig(
            skip_conv_layers=[],
            skip_dense_layer = False,
            nbit_input=8,
            nbit_weight=8,
            calibrate_mode="global_scale",
            global_scale=8.0,
            weight_scale="max",
            dtype_input='uint8',
            dtype_weight='int8',
            dtype_activation='int32',
            debug_enabled_ops=None)

mod_bert_int8 = quantize_relay_module(mod_bert_fp32, params_bert_fp32, qconfig=qconfig)

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# tune
measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(timeout=10),
	runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000))

target = "llvm -mcpu=cascadelake"
target_host = "llvm -mcpu=cascadelake"
with tvm.target.Target(target, host=target_host):
    with TempOpAttr(
        "nn.dense", "FTVMAlterOpLayout", topi.x86.dense_alter_op._alter_dense_layout
    ):
        res_func = run_opt_pass(mod_bert_int8['main'], transform.AlterOpLayout())
        res_func = run_opt_pass(res_func, transform.FoldConstant())

tasks = tvm.autotvm.task.extract_from_program(res_func, target="llvm -mcpu=cascadelake", params=params_bert_fp32)
print(tasks)

# for i, task in enumerate(reversed(tasks)):
#     prefix = "[Task %2d/%2d %s] " % (i + 1, len(tasks), task.name)
#     print(prefix, task)

#     n_trial = 1500
#     tsk_trial = min(n_trial, len(task.config_space))
#     early_stopping = 600

# 	# tuner = autotvm.tuner.RandomTuner(task)
#     tuner = autotvm.tuner.GATuner(task, pop_size=100)
#     tuner.tune(
# 		n_trial=tsk_trial,
#         early_stopping=early_stopping,
# 		measure_option=measure_option,
# 		callbacks=[
#             autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
#             autotvm.callback.log_to_file(log_file)
#         ],
# 	)


# evaluate
# target = "llvm -mcpu=cascadelake"
# with autotvm.apply_history_best(log_file):
#     print("Compile...")
#     with tvm.transform.PassContext(opt_level=3):
#         lib = relay.build(mod_bert_int8, target=target, params=params_bert_fp32)

# dev = tvm.cpu()
# tt_a = tvm.nd.array(tokens_tensor.numpy(), dev) #attention_mask
# st_a = tvm.nd.array(segments_tensors.numpy(), dev) #input_ids
# module = runtime.GraphModule(lib["default"](dev))
# module.set_input("input_ids", tvm.nd.array(tt_a))
# module.set_input("attention_mask", tvm.nd.array(st_a))

# # evaluate
# print("Evaluate inference time cost...")
# print(module.benchmark(dev, number=100, repeat=3))

# profile
# dev = tvm.cpu()
# target = "llvm -mcpu=cascadelake"
# with autotvm.apply_history_best(log_file):
#     print("Compile...")
#     with tvm.transform.PassContext(opt_level=3):
#         lib = relay.build(mod_bert_int8, target=target, params=params_bert_fp32)

# from tvm.contrib.debugger.debug_executor import GraphModuleDebug
# m = GraphModuleDebug(
#     lib["debug_create"]("default", dev),
#     [dev],
#     lib.graph_json,
#     dump_root="./profile_int8_bert_base_latency",
# )

