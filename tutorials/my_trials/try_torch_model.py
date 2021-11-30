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

mod_bert = relay.transform.CanonicalizeOps()(mod_bert)
mod_bert = relay.transform.InferType()(mod_bert)
mod_bert = relay.transform.SimplifyInference()(mod_bert)
mod_bert = relay.transform.FoldConstant()(mod_bert)
mod_bert = relay.transform.FoldScaleAxis()(mod_bert)
mod_bert = relay.transform.FoldConstant()(mod_bert)
print(mod_bert)

# mod_bert = relay.transform.MergeComposite(pattern_table())(mod_bert)
# mod_bert = relay.transform.AnnotateTarget(["dnnl"])(mod_bert)
# mod_bert = relay.transform.MergeCompilerRegions()(mod_bert)
# mod_bert = relay.transform.PartitionGraph()(mod_bert)

# target_host = 'llvm'
# target = 'llvm'
# ctx = tvm.cpu()

# tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx)
# st_a = tvm.nd.array(segments_tensors.numpy(), ctx)
# with tvm.transform.PassContext(opt_level=3):
#         graph, lib, params = tvm.relay.build(mod_bert,
#                                      target=target,
#                                      target_host=target_host,
#                                      params=params_bert)

# module = tvm.contrib.graph_executor.create(graph, lib, ctx)
# module.run()
