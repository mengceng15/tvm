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
# pylint: disable=no-else-return, invalid-name, unused-import
"""The layout auto-query func for dnnl."""

from . import _ffi_api

def AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW)  # type: ignore

def AutoQuery_matmul(M, K, N):
    return _ffi_api.AutoQuery_matmul(M, K, N)  # type: ignore

def AutoQuery_innerproduct(B, IC, OC):
    return _ffi_api.AutoQuery_innerproduct(B, IC, OC)  # type: ignore