# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
import tvm
from tvm import relay
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import tune_extracted_tasks
from tvm.meta_schedule.relay_integration import extract_task_from_relay
from tvm.meta_schedule import ApplyHistoryBest
from tvm.meta_schedule import schedule_rule, postproc
from tvm.meta_schedule.tune import Parse
from get_int8_bert import get_quantized_bert_base
from tvm import meta_schedule as ms
from tvm.tir.tensor_intrin import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN
# import tempfile
import tvm.topi.testing


config = ms.TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=32,
    max_trials_per_task=512,
    max_trials_global=20000,
)

sch_rules_for_vnni = [
    schedule_rule.AutoInline(
        into_producer=False,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=True,
        require_injective=True,
        require_ordered=True,
        disallow_op=["tir.exp"],
    ),
    schedule_rule.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
    schedule_rule.MultiLevelTilingWithIntrin(
        VNNI_INTRIN,
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="global",
        ),
    ),
    schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=16,
        max_vectorize_extent=64,
        unroll_max_steps=[0, 16, 64, 512],
        unroll_explicit=True,
    ),
    schedule_rule.RandomComputeLocation()
]

postprocs_for_vnni = [
    postproc.DisallowDynamicLoop(),
    postproc.RewriteParallelVectorizeUnroll(),
    postproc.RewriteReductionBlock(),
    postproc.RewriteTensorize(vectorize_init_loop=True),
]

def _test_bert_int8(target, sch_rules, postprocs):
    model_bs = 1
    if model_bs == 1:
        work_dir = "./tune_logs/latency"
    else:
        work_dir = "./tune_logs/throughput"
    relay_mod, params, input_info = get_quantized_bert_base(model_bs)

    relay_mod = relay.transform.FastMath()(relay_mod)

    ## tune
    # extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    # tune_tasks = []

    # for task in filter(
    #     lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
    #     extracted_tasks,
    # ):
    #     relay_func = list(task.mod.functions.values())[0]
    #     out_type = relay_func.body.checked_type

    #     if out_type.dtype != "float32":
    #         tune_tasks.append(task)

    # print(tune_tasks)

    # database = tune_extracted_tasks(
    #     tune_tasks,
    #     config,
    #     work_dir=work_dir,
    #     sch_rules=lambda: sch_rules,
    #     postprocs=lambda: postprocs,
    # )

    ## evaluate
    # database = Parse._database(None, path=work_dir)

    # dev = tvm.device("cuda" if "nvidia" in target else target, 0)

    # with ApplyHistoryBest(database):
    #     with tvm.transform.PassContext(
    #         opt_level=3,
    #         config={"relay.backend.use_meta_schedule": True},
    #     ):
    #         lib = relay.build(relay_mod, target=target, params=params)
    # runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # inputs = []
    # for name, shape in input_info:
    #     arr = np.random.uniform(1, 10, size=shape).astype("int64")
    #     runtime.set_input(name, arr)
    #     inputs.append(arr)

    # print(runtime.benchmark(dev, number=100, repeat=3).mean)

    #profile
    dev = tvm.cpu()

    database = Parse._database(None, path=work_dir)
    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    from tvm.contrib.debugger.debug_executor import GraphModuleDebug
    m = GraphModuleDebug(
        lib["debug_create"]("default", dev),
        [dev],
        lib.graph_json,
        dump_root="./profile_int8_bert_base_latency",
    )

    inputs = []

    for name, shape in input_info:
        arr = np.random.uniform(1, 10, size=shape).astype("int64")
        m.set_input(name, arr)
        inputs.append(arr)

    m.run()


@pytest.mark.skip("Requires cascadelake")
def test_vnni_bert_int8():
    _test_bert_int8("llvm -mcpu=cascadelake -model=platinum-8280 -num-cores 28", sch_rules_for_vnni, postprocs_for_vnni)

if __name__ == "__main__":
    test_vnni_bert_int8()
