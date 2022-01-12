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

from tvm import auto_scheduler

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
print(len(indexed_tokens))
print(len(segments_ids))

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
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
mod_bert, params_bert = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")

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

#copied from tlcbench
network_to_n_trials = {
    # CPU
    ("resnet_50", 1, "float32", "llvm"): 22000,
    ("mobilenet_v2", 1, "float32", "llvm"): 16000,
    ("bert", 1, "float32", "llvm"): 12000,
    # GPU
    ("resnet_50", 1, "float32", "cuda"): 20000,
    ("mobilenet_v2", 1, "float32", "cuda"): 16000,
    ("bert", 1, "float32", "cuda"): 12000,
}

def use_graph_tuner(network_name, batch_size, dtype, target):
    """Return whether use graph tuner for a network on a target"""
    # Only use graph tuner for CNNs on CPUs
    return "cpu" in target.keys and not (network_name in ["bert"])

def make_network_key(network_name, batch_size, dtype):
    return "%s-B%s-%s" % (network_name, batch_size, dtype)


def auto_scheduler_tune(network, batch_size, dtype, target, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)

    mod = mod_bert
    params = params_bert

    n_trials = network_to_n_trials[(network, batch_size, dtype, str(target.kind))]

    if "cpu" in target.keys:
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        min_repeat_ms = 450 if network in ["bert"] else 300
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=min_repeat_ms, timeout=10
        )
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_opt)

def benchmark(network, batch_size, dtype, target, log_file, repeat):
    mod = mod_bert
    params = params_bert

    assert os.path.exists(log_file), "The log file '%s' does not exist." % log_file
    print("Use log file %s" % log_file)

    # Build module
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)

    ctx = tvm.cpu(0)
    
    module = runtime.GraphModule(lib["default"](ctx))

    # Feed input data
    tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx) #attention_mask
    st_a = tvm.nd.array(segments_tensors.numpy(), ctx) #input_ids
    module.set_input("input_ids", tvm.nd.array(tt_a))
    module.set_input("attention_mask", tvm.nd.array(st_a))

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=repeat)
    return np.array(ftimer().results)

    # # profile
    # batches = 100
    # warmup = 500
    # with auto_scheduler.ApplyHistoryBest(log_file):
    #     with tvm.transform.PassContext(
    #         opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    #     ):
    #         json, lib, params = relay.build(mod, target=target, params=params)

    # ctx = tvm.cpu(0)

    # from tvm.contrib.debugger import debug_executor as graph_executor
    # rt_mod = graph_executor.create(json, lib, ctx)

    # tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx) #attention_mask
    # st_a = tvm.nd.array(segments_tensors.numpy(), ctx) #input_ids
    # rt_mod.set_input("input_ids", tvm.nd.array(tt_a))
    # rt_mod.set_input("attention_mask", tvm.nd.array(st_a))

    # for i in range(batches+warmup):
    #     tmp = rt_mod.profile()
    #     if i == batches + warmup - 1:
    #         print(tmp)


network = "bert"
batch_size = 1
dtype = "float32"
target = tvm.target.Target("llvm")
logdir = "tmp_logs/"

network_key = make_network_key(network, batch_size, dtype)
print("Tune %s ..." % network_key)

log_prefix = os.path.join(logdir, "auto_scheduler", target.model, network_key)
# auto_scheduler_tune(network, batch_size, dtype, target, log_prefix)

res = benchmark(network, batch_size, dtype, target, log_prefix, repeat=3)
message = "%-18s %-12s %-19s (%s)" % (
                    network,
                    batch_size,
                    "%.5f s" % np.mean(res),
                    "%.5f s" % np.std(res),
                )
print(message)