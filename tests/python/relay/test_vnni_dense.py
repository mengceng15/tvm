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
import sys
import numpy as np
import tvm
import tvm.testing
import tvm.contrib.graph_executor as runtime
from tvm import relay


def avx512_vnni_enabled():
    if sys.platform.startswith("linux"):
        return "avx512_vnni" in open("/proc/cpuinfo", "r").read()
    return False


def test_vnni_int8_dense(M, K, N):
    dev = tvm.cpu()

    x = relay.var('data', shape=(M, K), dtype="uint8")
    w = relay.var('weight', shape=(N, K), dtype="int8")
    out = relay.nn.dense(x, w, out_dtype="int32")
    mod = tvm.IRModule.from_expr(out)
    mod = relay.transform.InferType()(mod)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm -mcpu=cascadelake -model=platinum-8280")

    a_np = np.random.randint(low=-128, high=127, size=(M, K)).astype("uint8")
    b_np = np.random.randint(low=-128, high=127, size=(N, K)).astype("int8")
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)

    module = runtime.GraphModule(lib["default"](dev))
    module.set_input('data', a_tvm)
    module.set_input('weight', b_tvm)
    module.run()
    np_ans = np.matmul(a_np.astype("int32"), b_np.T.astype("int32"))

    tvm.testing.assert_allclose(module.get_output(0).numpy(), np_ans, atol=0, rtol=0)


def test_vnni_int8_dense_with_bias(M, K, N):
    dev = tvm.cpu()

    x = relay.var('data', shape=(M, K), dtype="uint8")
    w = relay.var('weight', shape=(N, K), dtype="int8")
    b = relay.var('bias', shape=(M, N), dtype="int32")
    out = relay.nn.dense(x, w, out_dtype="int32")
    out = relay.add(out, b)
    mod = tvm.IRModule.from_expr(out)
    mod = relay.transform.InferType()(mod)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm -mcpu=cascadelake -model=platinum-8280")

    llvm_ir = lib.lib.get_source("asm")
    assert "vpdpbusd" in llvm_ir, "VNNI is unused"

    a_np = np.random.randint(low=-128, high=127, size=(M, K)).astype("uint8")
    b_np = np.random.randint(low=-128, high=127, size=(N, K)).astype("int8")
    c_np = np.random.randint(low=-128, high=127, size=(M, N)).astype("int32")
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)

    module = runtime.GraphModule(lib["default"](dev))
    module.set_input('data', a_tvm)
    module.set_input('weight', b_tvm)
    module.set_input('bias', c_tvm)
    module.run()
    np_ans = np.matmul(a_np.astype("int32"), b_np.T.astype("int32")) + c_np

    tvm.testing.assert_allclose(module.get_output(0).numpy(), np_ans, atol=0, rtol=0)


if __name__ == "__main__":
    if avx512_vnni_enabled():
        test_vnni_int8_dense(128, 3072, 768)
        test_vnni_int8_dense(128, 768, 768)
        test_vnni_int8_dense(128, 768, 3072)
        test_vnni_int8_dense(1, 64, 128)
        test_vnni_int8_dense_with_bias(128, 3072, 768)
