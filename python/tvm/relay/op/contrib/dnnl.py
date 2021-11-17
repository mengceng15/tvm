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
# pylint: disable=invalid-name, unused-argument
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import tvm.ir
from ...dataflow_pattern import wildcard, is_op, is_expr
from .register import register_pattern_table
from tvm.relay.expr import const
import math

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


# _register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
# _register_external_op_helper("add")
# _register_external_op_helper("subtract")
# _register_external_op_helper("multiply")

_register_external_op_helper("concatenate")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.matmul")

def make_pattern(with_bias=True, with_relu=True):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    if with_relu:
        return is_op("nn.relu")(conv_out)
    return conv_out

def make_dense_pattern(with_bias=True, with_relu=False):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op("add")(dense, bias)
    else:
        dense_out = dense
    if with_relu:
        dense_out = is_op("nn.relu")(dense_out)
    return dense_out

def make_conv_add_sum_relu_pattern():
    data1 = wildcard()
    weight = wildcard()
    bias = wildcard()
    data2 = wildcard()
    out = is_op("nn.conv2d")(data1, weight)
    out = is_op("add")(out, bias)
    out = is_op("add")(out, data2)
    out = is_op("nn.relu")(out)
    return out

def make_matmul_pattern(with_bias=False, activation_type = "none"):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    matmul = is_op("nn.matmul")(data, weight)
    if with_bias:
        matmul_out = is_op("add")(matmul, bias)
    else:
        matmul_out = matmul
    if (activation_type == "relu") :
        matmul_out = is_op("nn.relu")(matmul_out)
    if (activation_type == "gelu") :
        const1 = is_expr(const(0.044715))
        const2 = is_expr(const(math.sqrt(2 / math.pi)))
        gelu = is_op("power")(matmul_out, is_expr(const(3, dtype="float32")))
        gelu = is_op("multiply")(gelu, const1)
        gelu = is_op("add")(gelu, matmul_out)
        gelu = is_op("multiply")(gelu, const2)
        gelu = is_op("tanh")(gelu)
        gelu = is_op("add")(gelu, is_expr(const(1, dtype="float32")))
        gelu = is_op("multiply")(gelu, is_expr(const(0.5)))
        matmul_out = is_op("multiply")(gelu, matmul_out)
    return matmul_out

def make_matmul_bias_mul_pattern():
    data1 = wildcard()
    weight = wildcard()
    bias = wildcard()
    data2 = wildcard()
    matmul = is_op("nn.matmul")(data1, weight)
    bias = is_op("add")(matmul, bias)
    out = is_op("multiply")(bias, data2)
    return out

@register_pattern_table("dnnl")
def pattern_table():
    conv2d_bias_sum_relu_pat = ("dnnl.conv2d_bias_sum_relu", make_conv_add_sum_relu_pattern())
    conv2d_bias_relu_pat = ("dnnl.conv2d_bias_relu", make_pattern(with_bias=True))
    conv2d_bias_pat = ("dnnl.conv2d_bias", make_pattern(with_bias=True, with_relu=False))
    dense_bias_relu_pat = ("dnnl.dense_bias_relu", make_dense_pattern(with_bias=True, with_relu=True))
    dense_bias_pat = ("dnnl.dense_bias", make_dense_pattern(with_bias=True))
    matmul_bias_relu_pat = ("dnnl.matmul_bias_relu", make_matmul_pattern(with_bias=True, activation_type="relu"))
    matmul_bias_gelu_pat = ("dnnl.matmul_bias_gelu", make_matmul_pattern(with_bias=True, activation_type="gelu"))
    matmul_bias_pat = ("dnnl.matmul_bias", make_matmul_pattern(with_bias=True))
    matmul_bias_mul_pat = ("dnnl.matmul_bias_mul", make_matmul_bias_mul_pattern())
    dnnl_patterns = [conv2d_bias_sum_relu_pat, conv2d_bias_relu_pat, conv2d_bias_pat, dense_bias_relu_pat,
     dense_bias_pat, matmul_bias_relu_pat, matmul_bias_gelu_pat, matmul_bias_mul_pat, matmul_bias_pat] #conv2d_relu_pat, 
    return dnnl_patterns
    
