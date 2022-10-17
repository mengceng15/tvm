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
#
"""Example code to do convolution."""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from tvm.topi.arm_cpu.conv2d_gemm import is_aarch64_arm
from tvm.topi.nn.conv2d import _get_workload
from tvm.topi.generic.conv2d import fallback_schedule_cpu_common_int8

from common import Int8Fallback
import tvm.testing
import pytest
import platform


def verify_conv2d_NCHWc_int8(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A", dtype="uint8")
    W = te.placeholder((num_filter, in_channel, kernel, kernel), name="W", dtype="int8")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    out_dtype = "int32"

    def check_target(target, compute, schedule, oc_block_factor):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        if target == "cuda" and not tvm.contrib.nvcc.have_int8(dev.compute_version):
            print("Skip because int8 intrinsics are not available")
            return

        @memoize("topi.tests.test_topi_conv2d_int8.verify_conv2d_nchw")
        def get_ref_data():
            a_np = np.random.randint(low=0, high=255, size=a_shape).astype(out_dtype)
            w_np = np.random.randint(low=-128, high=127, size=w_shape).astype(out_dtype)
            dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
            c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding).astype(
                out_dtype
            )

            # convert to NCHWc
            _, _, out_height, out_width = c_np.shape
            c_np = c_np.reshape(
                (batch, num_filter // oc_block_factor, oc_block_factor, out_height, out_width)
            ).transpose(0, 3, 4, 1, 2)

            return a_np, w_np, c_np

        a_np, w_np, c_np = get_ref_data()

        with tvm.target.Target(target):
            C = compute(
                A,
                W,
                (stride, stride),
                padding,
                (dilation, dilation),
                "NCHW",
                "NCHW",
                out_dtype,
            )
            s = schedule([C])

        print("A shape: ", A.shape)
        print("W shape: ", W.shape)
        print("C shape: ", C.shape)

        #print(tvm.lower(s, [A, W, C], simple_mode=True))

        a = tvm.nd.array(a_np.astype("uint8"), dev)
        w = tvm.nd.array(w_np.astype("int8"), dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)

        compile_args = [A, W, C]
        run_args = [a, w, c]

        func = tvm.build(
            s,
            compile_args,
            target,
            name="%d_%d_%d_%d_%d_%d_%d_%d"
            % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
        )

        print("Running on target: %s" % target)

        func(*run_args)

        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)
        print("pass")
        #print("c.numpy(): ", c.numpy())
        #print("c.np: ", c_np)
    
    targets = []
    targets.append(
        (
            "llvm -mcpu=cascadelake -model=platinum-8260",
            topi.x86.conv2d_NCHWc_int8,
            topi.x86.schedule_conv2d_NCHWc_int8,
            16,
        )
    )

    for target, compute, schedule, oc_block_factor in targets:
        check_target(target, compute, schedule, oc_block_factor)

def test_conv2d_nchw():
    verify_conv2d_NCHWc_int8(1, 128, 64, 128, 3, 1, 0)
    #verify_conv2d_NCHWc_int8(1, 4, 3, 16, 3, 1, 0)

if __name__ == "__main__":
    test_conv2d_nchw()
