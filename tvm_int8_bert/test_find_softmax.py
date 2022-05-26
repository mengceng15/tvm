import numpy as np

import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay
import tvm.contrib.graph_executor as runtime
from tvm.relay.testing import *


# # mod_bert_int8 = quantize_relay_module(mod_bert_fp32, params_bert_fp32, qconfig=qconfig)
# class FindSoftmax(ExprVisitor):
#     """
#     Visits the Graph recursively and checks if it contains ops in the op_list
#     """

#     def __init__(self):
#         ExprVisitor.__init__(self)

#     def visit_call(self, call):
#         if "softmax" in call.op.name:
#             print(type(call))
#             print(call.attrs.axis)

#         return super().visit_call(call)

# f = FindSoftmax()
# f.visit(mod_bert_fp32["main"].body)

target = "llvm -mcpu=cascadelake"

dev = tvm.cpu()

tensor_x = np.random.rand(1, 12, 128, 128).astype("float32")
x_tvm = tvm.nd.array(tensor_x, dev)

x = relay.var('x', shape=(1, 12, 128, 128))
out1 = relay.nn.softmax(x, axis=-1)
out2 = relay.nn.softmax(x, axis=3)
out = relay.Tuple([out1, out2])
f = relay.Function([x], out)
softmax = tvm.IRModule()
softmax["main"] = f

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(softmax, target=target)

module = runtime.GraphModule(lib["default"](dev))
module.set_input("x", tvm.nd.array(x_tvm))
module.run()
output0 = module.get_output(0).numpy()
output1 = module.get_output(1).numpy()
print(output0.shape)
print(output1.shape)

np.testing.assert_allclose(output0, output1, rtol=1e-5, atol=0)


