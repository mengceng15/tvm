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
"""Example code to do vnni int8 dense."""
import sys
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple
import tvm.testing


def avx512_vnni_enabled():
    if sys.platform.startswith("linux"):
        return "avx512_vnni" in open("/proc/cpuinfo", "r").read()
    return False


def verify_dense_vnni_int8(M, K, N):
    A = te.placeholder((M, K), name="A", dtype="uint8")
    W = te.placeholder((N // 16, K // 4, 16, 4), name="W", dtype="int8")
    B = te.placeholder((M, N), name="bias", dtype="int32")

    @memoize("topi.tests.test_topi_dense_vnni_int8.verify_vnni_int8")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=(M, K)).astype("uint8")
        w_np = np.random.randint(low=-128, high=127, size=(N, K)).astype("int8")
        b_np = np.random.randint(low=-128, high=127, size=(M, N)).astype("int32")
        c1_np = np.matmul(a_np.astype("int32"), w_np.T.astype("int32"))
        c2_np = c1_np + b_np
        return a_np, w_np, b_np, c1_np, c2_np

    a_np, w_np, b_np, c1_np, c2_np = get_ref_data()

    def check_target(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            fcompute = topi.x86.dense_vnni
            fschedule = topi.x86.schedule_dense_vnni
            t_gemm = fcompute(A, W)
            t_bias = topi.add(t_gemm, B)
            s1 = fschedule([t_gemm])
            s2 = fschedule([t_bias])
        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np.reshape(N, K//4, 4).transpose(1, 2, 0).reshape(K // 4, 4, N // 16, 16).transpose(2, 0, 3, 1), dev)
        b = tvm.nd.array(b_np, dev)

        gemm_out = tvm.nd.array(np.zeros(get_const_tuple(t_gemm.shape), dtype=t_gemm.dtype), dev)
        bias_out = tvm.nd.array(np.zeros(get_const_tuple(t_bias.shape), dtype=t_bias.dtype), dev)
        
        func1 = tvm.build(s1, [A, W, t_gemm], target)
        func2 = tvm.build(s2, [A, W, B, t_bias], target)
        assert "vpdpbusd" in func1.get_source("asm")
        assert "vpdpbusd" in func2.get_source("asm")
        func1(a, w, gemm_out)
        func2(a, w, b, bias_out)
        tvm.testing.assert_allclose(gemm_out.numpy(), c1_np, rtol=1e-5)
        tvm.testing.assert_allclose(bias_out.numpy(), c2_np, rtol=1e-5)

    check_target("llvm -mcpu=cascadelake -model=platinum-8280")


def test_dense_vnni_int8():
    if avx512_vnni_enabled():
        verify_dense_vnni_int8(128, 3072, 768)
        verify_dense_vnni_int8(128, 768, 768)
        verify_dense_vnni_int8(128, 768, 3072)
        verify_dense_vnni_int8(1, 64, 128)
        print("passed")
    else:
        print("avx512_vnni is unsupported")


if __name__ == "__main__":
    test_dense_vnni_int8()
