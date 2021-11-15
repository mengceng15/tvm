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
def use_graph_tuner(network_name, batch_size, dtype, target):
    """Return whether use graph tuner for a network on a target"""
    # Only use graph tuner for CNNs on CPUs
    return "cpu" in target.keys and not (network_name in ["bert"])

def make_network_key(network_name, batch_size, dtype):
    return "%s-B%s-%s" % (network_name, batch_size, dtype)

def autotvm_tune(network, batch_size, dtype, target, log_prefix):
    kernel_log = log_prefix + ".kernel.log"
    graph_log = log_prefix + ".graph.log"
    os.makedirs(os.path.dirname(graph_log), exist_ok=True)
    if os.path.exists(kernel_log):
        os.remove(kernel_log)
    if os.path.exists(graph_log):
        os.remove(graph_log)

    layout = "NCHW"
    # mod, params, input_name, input_shape, output_shape = get_network(
    #     network, batch_size, dtype, layout
    # )
    mod = mod_bert
    params = params_bert
    tuning_opt = get_tuning_option(network, batch_size, dtype, target, kernel_log)
    ops = [
        relay.op.get("nn.batch_matmul"),
        relay.op.get("nn.dense"),
        relay.op.get("nn.conv2d"),
    ]

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=ops
    )
    tune_kernels(tasks, **tuning_opt)

    if use_graph_tuner(network, batch_size, dtype, target):
        tune_graph(mod["main"], input_name, input_shape, target, kernel_log, graph_log)


def get_tuning_option(network, batch_size, dtype, target, log_file):
    if "cpu" in target.keys:
        if use_graph_tuner(network, batch_size, dtype, target):
            tuning_option = {
                "log_filename": log_file,
                "tuner": "random",
                "n_trial": 1300,
                "early_stopping": None,
                "use_transfer_learning": False,
                "measure_option": autotvm.measure_option(
                    builder=autotvm.LocalBuilder(timeout=10),
                    runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
                ),
            }
        else:
            # tuning_option = {
            #     "log_filename": log_file,
            #     "tuner": "xgb",
            #     "n_trial": 1500,
            #     "early_stopping": 600,
            #     "use_transfer_learning": True,
            #     "measure_option": autotvm.measure_option(
            #         builder=autotvm.LocalBuilder(timeout=10),
            #         runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
            #     ),
            # }
            tuning_option = {
                "log_filename": log_file,
                "tuner": "random",
                "n_trial": 1300,
                "early_stopping": None,
                "use_transfer_learning": False,
                "measure_option": autotvm.measure_option(
                    builder=autotvm.LocalBuilder(timeout=10),
                    runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
                ),
            }
    else:
        tuning_option = {
            "log_filename": log_file,
            "tuner": "xgb",
            "n_trial": 2000,
            "early_stopping": 600,
            "use_transfer_learning": True,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(
                    number=20, repeat=3, timeout=4, min_repeat_ms=150
                ),
            ),
        }

    return tuning_option


def tune_kernels(
    tasks,
    measure_option,
    tuner,
    n_trial,
    early_stopping,
    log_filename,
    use_transfer_learning,
):
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "random" or n_trial >= len(tsk.config_space):
            tuner_obj = RandomTuner(tsk)
        elif tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
            # use history data to pre-train the cost model
            if use_transfer_learning:
                if os.path.isfile(log_filename):
                    tuner_obj.load_history(autotvm.record.load_from_file(log_filename))
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(
    graph, input_name, input_shape, target, kernel_log, graph_log, use_DP=True
):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: input_shape}, kernel_log, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(graph_log)

def benchmark(network, batch_size, dtype, target, log_prefix, repeat):
    mod = mod_bert
    params = params_bert

    log_file = log_prefix + ".kernel.log"
    history_best_context = autotvm.apply_history_best(log_file)

    assert os.path.exists(log_file), "The log file '%s' does not exist." % log_file
    print("Use log file %s" % log_file)

    # Build module
    with history_best_context:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    ctx = tvm.cpu()
    module = runtime.GraphModule(lib["default"](ctx))

    # # Feed input data
    tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx) #attention_mask
    st_a = tvm.nd.array(segments_tensors.numpy(), ctx) #input_ids
    module.set_input("input_ids", tvm.nd.array(tt_a))
    module.set_input("attention_mask", tvm.nd.array(st_a))

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=repeat)
    return np.array(ftimer().results)

network = "bert"
batch_size = 1
dtype = "float32"
target = tvm.target.Target("llvm")
logdir = "tmp_logs/"

network_key = make_network_key(network, batch_size, dtype)
print("Tune %s ..." % network_key)

log_prefix = os.path.join(logdir, "autotvm", target.model, network_key)
# autotvm_tune(network, batch_size, dtype, target, log_prefix)
res = benchmark(network, batch_size, dtype, target, log_prefix, repeat=3)
message = "%-18s %-12s %-19s (%s)" % (
                    network,
                    batch_size,
                    "%.5f s" % np.mean(res),
                    "%.5f s" % np.std(res),
                )
print(message)