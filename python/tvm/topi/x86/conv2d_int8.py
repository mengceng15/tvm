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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
# pylint: disable=no-value-for-parameter,import-outside-toplevel
"""Conv2D int8 schedule on x86"""

import tvm
from tvm import te
from tvm import autotvm
from ..nn.conv2d import _get_workload as _get_conv2d_workload
from .. import tag
from ..generic import conv2d as conv2d_generic
from ..nn.utils import get_pad_tuple
from ..nn.conv2d import unpack_NCHWc_to_nchw
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..utils import get_const_tuple, traverse_inline
from .. import nn
from . import conv2d_avx_1x1, conv2d_avx_common
from .utils import target_has_sse42
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity


def fallback_schedule_x86_int8(cfg, wkl, int32_lanes, num_int8_elements):
    """Fallback schedule for conv2d int8 on x86 cpu.
    Normally the inner most pattern takes two int8/uint8 tensors
    data[num_int8_elements] and kernel[int32_lanes, num_int8_elements],
    produces a dot product int32/uint32 output[int32_lanes].

    Parameters
    ----------
    int32_lanes : int
        How many numbers of int32/uint32 will be produced using intrinsic.
        This is related to output channel.
    num_int8_elements : int
        How many numbers of input int32/uint32 will be multiplied and reduced.
        This is related to input channel.
    """
    pt, pl, pb, pr = wkl.padt, wkl.padl, wkl.padb, wkl.padr
    HSTR, WSTR = wkl.stride_h, wkl.stride_w
    dilated_kernel_w = (wkl.kernel_w - 1) * wkl.dilation_w + 1
    out_width = (wkl.width + pl + pr - dilated_kernel_w) // WSTR + 1

    assert wkl.out_filter % int32_lanes == 0, "wkl.out_filter=%d, int32_lanes=%d" % (
        wkl.out_filter,
        int32_lanes,
    )
    assert wkl.in_filter % num_int8_elements == 0, "wkl.in_filter=%d, num_int8_elements=%d" % (
        wkl.in_filter,
        num_int8_elements,
    )

    oc_bn = int32_lanes if int32_lanes >= num_int8_elements else num_int8_elements
    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    oh_bn = 1
    for n in range(31, 0, -1):
        if oh_bn % n == 0:
            oh_bn = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["tile_oh"] = SplitEntity([-1, 1])


def fallback_schedule_x86_1x1_int8(cfg, wkl, int32_lanes, num_int8_elements):
    """Fallback schedule for 1x1 conv2d int8 on x86 cpu.
    Normally the inner most pattern takes two int8/uint8 tensors
    data[num_int8_elements] and kernel[int32_lanes, num_int8_elements],
    produces a dot product int32/uint32 output[int32_lanes].

    Parameters
    ----------
    int32_lanes : int
        How many numbers of int32/uint32 will be produced using intrinsic.
        This is related to output channel.
    num_int8_elements : int
        How many numbers of input int32/uint32 will be multiplied and reduced.
        This is related to input channel.
    """
    pt, pl, pb, pr = wkl.padt, wkl.padl, wkl.padb, wkl.padr
    HSTR, WSTR = wkl.stride_h, wkl.stride_w
    out_height = (wkl.height + pt + pb - wkl.kernel_h) // HSTR + 1
    out_width = (wkl.width + pl + pr - wkl.kernel_w) // WSTR + 1

    assert wkl.out_filter % int32_lanes == 0, "wkl.out_filter=%d, int32_lanes=%d" % (
        wkl.out_filter,
        int32_lanes,
    )
    assert wkl.in_filter % num_int8_elements == 0, "wkl.in_filter=%d, num_int8_elements=%d" % (
        wkl.in_filter,
        num_int8_elements,
    )

    oc_bn = int32_lanes if int32_lanes >= num_int8_elements else num_int8_elements
    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break

    for ow_factor in range(out_width, 0, -1):
        if out_width % ow_factor == 0:
            for oh_factor in range(out_height, 0, -1):
                if out_height % oh_factor == 0 and ow_factor * oh_factor < 32:
                    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
                    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
                    cfg["tile_oh"] = SplitEntity([out_height // oh_factor, oh_factor])
                    cfg["tile_ow"] = SplitEntity([out_width // ow_factor, ow_factor])
                    return
    raise ValueError("cannot decide default schedule for workload: {}".format(wkl))


def _get_default_config_int8(
    cfg,
    data,
    kernel,
    strides,
    padding,
    dilation,
    out_dtype,
    is_depthwise=False,
    layout="NCHW",
    int32_lanes=4,
):
    """
    Get default schedule config for the workload
    """
    if is_depthwise:
        # Fallback to FP32 default config until a VNNI schedule is defined.
        wkl = _get_depthwise_conv2d_workload(data, kernel, strides, padding, out_dtype)
        from .depthwise_conv2d import _fallback_schedule

        _fallback_schedule(cfg, wkl)
    else:
        wkl = _get_conv2d_workload(data, kernel, strides, padding, dilation, out_dtype, layout)
        is_kernel_1x1 = wkl.kernel_h == 1 and wkl.kernel_w == 1
        if is_kernel_1x1:
            fallback_schedule_x86_1x1_int8(
                cfg, wkl, int32_lanes=int32_lanes, num_int8_elements=4
            )
        else:
            fallback_schedule_x86_int8(
                cfg, wkl, int32_lanes=int32_lanes, num_int8_elements=4
            )


def is_int8_hw_support(data_dtype, kernel_dtype):
    """
    Checks to ensure that we can use Intel DLBoost instructions
    1) The datatypes are correct.
    2) LLVM version has support for the instructions.
    3) Target is skylake and above.
    """
    # 1) Check datatypes
    is_dtype_support = data_dtype == "uint8" and kernel_dtype == "int8"

    # 2) Check LLVM support
    llvm_version = tvm.target.codegen.llvm_version_major()
    is_llvm_support = llvm_version >= 8

    # 3) Check target
    mcpu = tvm.target.Target.current().mcpu
    is_target_support = target_has_sse42(mcpu)

    return is_dtype_support and is_llvm_support and is_target_support


def conv2d_nchw_int8(data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NCHW layout and int8 dtype"""
    layout = "NCHW"
    packed_out = conv2d_NCHWc_int8(
        data, kernel, strides, padding, dilation, layout, layout, out_dtype
    )
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def schedule_conv2d_nchw_int8(outs):
    """Create the schedule for conv2d_nchw_int8"""
    return schedule_conv2d_NCHWc_int8(outs)


def _pack_data(cfg, data, kernel):
    n_elems = 4
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    #nchw -> nhwcc
    data = te.compute(
        (n, ih, iw, ic_chunk, ic_bn),
        lambda bs, h, w, c, vc: data[bs, c * ic_bn + vc, h, w],
        name="data_vec",
    )

    # oihw -> occ kh kw icc icbc ocb icbb
    kernel = te.compute(
        (oc_chunk, kh, kw, ic_chunk, ic_bn // n_elems, oc_bn, n_elems),
        lambda occ, k_h, k_w, icc, icbc, ocb, icbb: kernel[
            occ * oc_bn + ocb, icc * ic_bn + icbc * n_elems + icbb, k_h, k_w
        ],
        name="kernel_vec",
    )

    return data, kernel


@autotvm.register_topi_compute("conv2d_NCHWc_int8.x86")
def conv2d_NCHWc_int8(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d with NCHWc layout and int8 dtype"""
    if len(data.shape) == 5:
        n, ih, iw, ic_chunk, ic_bn = get_const_tuple(data.shape)
        in_channel = ic_chunk * ic_bn
        oc_chunk, kernel_height, kernel_width, ic_chunk_group, _, oc_bn, _ = get_const_tuple(
            kernel.shape
        )
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    # Define autotvm tuning space
    is_kernel_1x1 = kernel_height == 1 and kernel_width == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kernel_height, kernel_width))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    dilated_kernel_h = (kernel_height - 1) * dh + 1
    dilated_kernel_w = (kernel_width - 1) * dw + 1
    oh = (ih - dilated_kernel_h + pt + pb) // sh + 1
    ow = (iw - dilated_kernel_w + pl + pr) // sw + 1

    cfg.define_split("tile_ic", in_channel, num_outputs=2, filter=lambda y: y.size[-1] % 4 == 0)
    cfg.define_split("tile_oc", num_filter, num_outputs=2, filter=lambda y: y.size[-1] % 16 == 0)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_split("tile_oh", oh, num_outputs=2)
    else:
        cfg.define_split("tile_oh", oh, num_outputs=2)

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config_int8(
            cfg,
            te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
            te.placeholder(
                (num_filter, in_channel, kernel_height, kernel_width), dtype=kernel.dtype
            ),
            strides,
            padding,
            dilation,
            out_dtype,
            int32_lanes=16,
        )

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        data, kernel = _pack_data(cfg, data, kernel)

    return nn.conv2d_NCHWc_int8(
        data, kernel, strides, padding, dilation, layout, out_layout, out_dtype
    )


@autotvm.register_topi_schedule("conv2d_NCHWc_int8.x86")
def schedule_conv2d_NCHWc_int8(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        """Traverse operators from computation graph"""
        if "conv2d_NCHWc_int8" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]

            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            # int8 conv kernel is 7-dim
            _, kh, kw, _, _, _, _ = get_const_tuple(kernel_vec.shape)
            if kh == 1 and kw == 1:
                conv2d_avx_1x1._schedule_conv_NCHWc_int8(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc_int8(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_schedule("conv2d_nhwc_pack_int8.x86")
def schedule_conv2d_nhwc_pack_int8(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else:  # inject custom schedule
                if len(op.axis) == 4:  # schedule bias + bn + relu
                    n, h, w, c = op.axis
                    fused = s[op].fuse(n, h, w)
                    s[op].parallel(fused)
                    s[op].vectorize(c)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if "conv2d_nhwc_pack_int8" in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = (
                data_vec.op.input_tensors[0]
                if isinstance(data_vec.op, te.tensor.ComputeOp) and "pad" not in data_vec.op.tag
                else data_vec
            )
            if isinstance(data.op, te.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, cfg, data_vec, conv_out, outs[0]]
            if data.dtype == "uint8":
                kh, kw, _, _, _ = get_const_tuple(kernel.shape)
                if kh == 1 and kw == 1:
                    conv2d_avx_1x1._schedule_conv_nhwc_pack_int8(*args)
                else:
                    raise ValueError("Only support 1x1 kernel with " "schedule_conv2d_nhwc_pack.")
            else:
                raise ValueError(
                    "Not support this data type {} with "
                    "schedule_conv2d_nhwc_pack. Only support int8".format(data.dtype)
                )

        scheduled_ops.append(op)

    traverse(output_op)
    return s
