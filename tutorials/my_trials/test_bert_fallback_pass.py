import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay
import tvm.contrib.graph_executor as runtime
from tvm.relay.testing import *
import os
from tvm.contrib import utils
from tvm.relay import nn

from transformers import BertModel, BertTokenizer, BertConfig
import torch
import numpy as np

from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner

from tvm.relay.expr import Call

from tvm.relay.testing.temp_op_attr import TempOpAttr

from tvm.topi.utils import get_const_tuple

enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)
torch_result = traced_model(tokens_tensor, segments_tensors)

shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
mod_bert, params_bert = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")

weight_dic = {"A":"N",
              "B":"C",
              "C":"H",
              "D":"W",
              "a":"n",
              "b":"c",
              "c":"h",
              "d":"w"}

@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    def transform_function(self, func, mod, ctx):
        class FallbackSpecialMatmul(tvm.relay.ExprMutator):
            def visit_call(self, call):
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                if "nn.special_matmul" == call.op.name:
                    in0_shape = call.type_args[0].shape
                    in1_shape = call.type_args[1].shape
                    if len(in0_shape) == 4 and len(in1_shape) == 4: # batch_matmul
                        in0_reshape = relay.reshape(new_args[0], (-1, in0_shape[-2], in0_shape[-1]))
                        trans_axes = list(range(len(in1_shape)))
                        trans_axes[-2], trans_axes[-1] = trans_axes[-1], trans_axes[-2]
                        in1_trans = relay.transpose(new_args[1], trans_axes)
                        in1_reshape = relay.reshape(in1_trans, (-1, in1_shape[-1], in1_shape[-2]))
                        batch_matmul = nn.batch_matmul(in0_reshape, in1_reshape)
                        batch_matmul_reshape = relay.reshape(batch_matmul, (*in0_shape[:-2], in0_shape[-2], in1_shape[-1]))
                        return batch_matmul_reshape
                    else: # in0->3d in1->2d
                        in0_reshape = relay.reshape(new_args[0], (-1, in0_shape[-1]))
                        matmul = nn.dense(in0_reshape, new_args[1])
                        matmul_reshape = relay.reshape(matmul, (*in0_shape[:-1], in1_shape[-2]))
                        return matmul_reshape
                return Call(new_fn, new_args, call.attrs, call.type_args, call.span)
        return FallbackSpecialMatmul().visit(func)

mod_bert = relay.transform.CanonicalizeOps()(mod_bert)
mod_bert = relay.transform.InferType()(mod_bert)
mod_bert = relay.transform.SimplifyInference()(mod_bert)

# enable the fallback pass if not using byoc onednn
custom_pass = CustomPipeline()
mod_bert = custom_pass(mod_bert)

mod_bert = relay.transform.FoldConstant()(mod_bert)
mod_bert = relay.transform.FoldScaleAxis()(mod_bert)
mod_bert = relay.transform.FoldConstant()(mod_bert)
mod_bert = relay.transform.FoldConstant()(mod_bert)
# print(mod_bert)

target = 'llvm'
ctx = tvm.cpu()

# Build module
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod_bert, target=target, params=params_bert)

module = tvm.contrib.graph_executor.create(graph, lib, ctx)

# Feed input data
tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx) #attention_mask
st_a = tvm.nd.array(segments_tensors.numpy(), ctx) #input_ids
module.set_input("input_ids", tvm.nd.array(tt_a))
module.set_input("attention_mask", tvm.nd.array(st_a))
module.set_input(**params)
module.run()

tvm_output_0 = module.get_output(0).numpy()
tvm_output_1 = module.get_output(1).numpy()
np.testing.assert_allclose(torch_result[0], tvm_output_0, rtol=1e-05, atol=1e-05)
np.testing.assert_allclose(torch_result[1], tvm_output_1, rtol=1e-05, atol=1e-05)
print("BYOC bert fallback pass correctness: PASSED")

    