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
from tvm.relay.op.nn.nn import special_matmul
from ...dataflow_pattern import wildcard, is_op, is_expr
from .register import register_pattern_table
from tvm.relay.expr import const

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


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("add")
_register_external_op_helper("multiply")
# debug
_register_external_op_helper("nn.special_matmul")


def make_pattern(with_bias=True):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    return is_op("nn.relu")(conv_out)

def make_specialmatmul_biasadd_pattern():
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    matmul = is_op("nn.special_matmul")(data, weight)
    bias_add = is_op("add")(matmul, bias)
    return bias_add

def make_specialmatmul_biasadd_gelu_pattern():
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    matmul = is_op("nn.special_matmul")(data, weight)
    bias_add = is_op("add")(matmul, bias)
    const1 = is_expr(const(1.41421))
    const2 = is_expr(const(0.5))
    const3 = is_expr(const(1.0))
    div = is_op("divide")(bias_add, const1)
    erf = is_op("erf")(div)
    mul = is_op("multiply")(bias_add, const2)
    add = is_op("add")(erf, const3)
    mul2 = is_op("multiply")(mul, add)
    return mul2

@register_pattern_table("dnnl")
def pattern_table():
    conv2d_bias_relu_pat = ("dnnl.conv2d_bias_relu", make_pattern(with_bias=True))
    conv2d_relu_pat = ("dnnl.conv2d_relu", make_pattern(with_bias=False))
    specialmatmul_biasadd_pat = ("dnnl.specialmatmul_biasadd",
     make_specialmatmul_biasadd_pattern())
    specialmatmul_biasadd_gelu_pat = ("dnnl.specialmatmul_biasadd_gelu",
     make_specialmatmul_biasadd_gelu_pattern())
    dnnl_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat, 
     specialmatmul_biasadd_gelu_pat, specialmatmul_biasadd_pat]
    return dnnl_patterns
