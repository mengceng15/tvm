from transformers import BertModel, BertTokenizer, BertConfig
import torch

import logging

import tvm

def get_quantized_bert_base(bs):
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

    batch_size = bs

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens] * batch_size)
    segments_tensors = torch.tensor([segments_ids] * batch_size)

    input_info = []
    input_info.append(("input_ids", tokens_tensor.numpy().shape))
    input_info.append(("attention_mask", segments_tensors.numpy().shape))

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

    return (mod_bert_int8, params_bert_fp32, input_info)