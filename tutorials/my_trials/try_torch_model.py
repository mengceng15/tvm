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
#torch.jit.save(traced_model, "/home/mengceng/traced_bert.pt")

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
                print(call.op.name)
                if "nn.special_matmul" == call.op.name:
                    in0_shape = call.type_args[0].shape
                    in1_shape = call.type_args[1].shape
                    if len(in0_shape) == 4 and len(in1_shape) == 4: # batch_matmul
                        print(in0_shape)
                        print(in1_shape)
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

mod_bert = relay.transform.CanonicalizeOps()(mod_bert)
mod_bert = relay.transform.InferType()(mod_bert)
mod_bert = relay.transform.SimplifyInference()(mod_bert)

# enable the fallback pass if not using byoc onednn
# custom_pass = CustomPipeline()
# print(mod_bert)
# mod_cus = custom_pass(mod_bert)
# print(mod_cus)

mod_bert = relay.transform.FoldConstant()(mod_bert)
mod_bert = relay.transform.FoldScaleAxis()(mod_bert)
mod_bert = relay.transform.FoldConstant()(mod_bert)
with TempOpAttr("nn.special_matmul", "FTVMAlterOpLayout", alter_special_matmul):
    mod_bert = relay.transform.AlterOpLayout()(mod_bert)
mod_bert = relay.transform.FoldConstant()(mod_bert)
mod_bert = relay.transform.MergeComposite(pattern_table())(mod_bert)
# print(mod_bert)
mod_bert = relay.transform.AnnotateTarget(["dnnl"])(mod_bert)
mod_bert = relay.transform.MergeCompilerRegions()(mod_bert)
mod_bert = relay.transform.PartitionGraph()(mod_bert)
# print(mod_bert)

target_host = 'llvm'
target = 'llvm'
ctx = tvm.cpu()

tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx) #attention_mask
st_a = tvm.nd.array(segments_tensors.numpy(), ctx) #input_ids
with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = tvm.relay.build(mod_bert,
                                     target=target,
                                     target_host=target_host,
                                     params=params_bert)

module = tvm.contrib.graph_executor.create(graph, lib, ctx)
module.set_input("attention_mask", tvm.nd.array(tt_a))
module.set_input("input_ids", tvm.nd.array(st_a))

import time

def x():
    for i in range(100):
        module.run()
    ctx.sync()

start = time.time()
x()
end = time.time()
print("time:", (end-start)/100)
