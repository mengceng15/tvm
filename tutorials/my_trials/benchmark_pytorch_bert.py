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

import time

def x():
    for i in range(1000):
        traced_model(tokens_tensor, segments_tensors)

start = time.time()
x()
end = time.time()
print("time:", (end-start)/1000)
