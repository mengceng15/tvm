'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
import mxnet as mx
import gluoncv

import warnings
warnings.filterwarnings("ignore")
import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay

import numpy as np
import os
from tvm.contrib import utils
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.download import download_testdata
from PIL import Image

cnt_conv_num = 0
network_dict = {"resnet18":"ResNet18_v1b",
                "resnet34":"ResNet34_v1b",
                "resnet50":"ResNet50_v1b",
                "resnet101":"ResNet101_v1b",
                "resnet152":"ResNet152_v1b",
                "vgg11":"VGG11",
                "vgg13":"VGG13",
                "vgg16":"VGG16",
                "vgg19":"VGG19",
                "vgg11_bn":"VGG11_bn",
                "vgg13_bn":"VGG13_bn",
                "vgg16_bn":"VGG16_bn",
                "vgg19_bn":"VGG19_bn",
                "densenet121":"DenseNet121",
                "InceptionV3":"InceptionV3",}

data_dic = {"a":"N",
            "b":"C",
            "c":"H",
            "d":"W",}

weight_dic = {"a":"O",
              "b":"I",
              "c":"H",
              "d":"W",}

translate_dict = {"abcd":"NCHW",
                "Acdb8a": "OHWI8o",
                "Acdb16a": "OHWI16o",
                "ABcd8b8a": "OIHW8i8o",
                "ABcd16b16a": "OIHW16i16o",}

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """Simple test function to replace one argument to another."""

    def __init__(self):
        self.cnt = 0
        self.net_dict = {}
        self.branch_post_op_dict = {} # dict for querying the post ops of the key node
        self.branchid = {}
        self.merge_dict = {}
        self.net_lst = []
        self.flag_for_merge_add = False

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        self.merge_consecutive_add(func.body)
        if not self.flag_for_merge_add:
            print("no consecutive add")
            return func
        res = self.rewrite_graph()
        res = relay.Function([self.input], res)
        return res

    def merge_consecutive_add(self, node):
        op_set, traversal_lst, tmp_op_lst = set(), [], []
        cnt_branch = 0
        traversal_lst.append(node)
        while traversal_lst:
            u = traversal_lst.pop()
            if u not in op_set and not self.check_constant(u):
                tmp_op_lst.append(self.cnt)
                self.net_dict[self.cnt] = u
                if self.check_Var(u):
                    self.input = u
                    self.net_lst = [tmp_op_lst]
                    tmp_op_lst = []
                    self.cnt += 1
                    continue
            else:
                if u in op_set:
                    cnt_branch += 1
                    pre_op_idx = [key for key, value in self.net_dict.items() if value == u][0]
                    self.branch_post_op_dict[pre_op_idx] = [pre_op_idx-1]
                    self.net_lst.append(tmp_op_lst)
                    if len(tmp_op_lst)!=0:
                        self.branch_post_op_dict[pre_op_idx].append(tmp_op_lst[-1])
                        self.branchid[tmp_op_lst[-1]] = cnt_branch
                        tmp_op_lst = []
                continue
            op_set.add(u)

            if self.check_consecutive_add(u):
                a1 = u
                a2 = a1.args[0]
                conv = a2.args[0]
                data = relay.add(a1.args[1], a2.args[1])
                traversal_lst.extend([data, conv])
                self.flag_for_merge_add = True
            else:
                if 'Tuple' not in str(type(u)):
                    traversal_lst.extend(list(u.args)[::-1])
                else:
                    for i in range(len(u)-1, -1, -1):
                        traversal_lst.append(u[i])
            if self.check_branch(u):
                if 'Tuple' not in str(type(u)):
                    self.merge_dict[self.cnt] = list(u.args)
                else:
                    concat_lst = []
                    for i in range(len(u)):
                        concat_lst.append(u[i])
                    self.merge_dict[self.cnt] = concat_lst
            self.cnt += 1
        
        ivd = dict((v, k) for k, v in self.net_dict.items())
        for key, value in self.merge_dict.items():
            tmp_lst = []
            for v in value:
                tmp_lst.append(ivd[v])
            self.merge_dict[key] = tmp_lst

    def rewrite_graph(self):
        main_idx = self.net_lst[0][-2]
        while main_idx>=0:
            if main_idx not in self.merge_dict.keys() and main_idx not in self.branch_post_op_dict.keys():
                pre_node = self.net_dict[main_idx+1]
                new_node = self.net_dict[main_idx]
                new_node = self.get_op(new_node, pre_node)
                self.net_dict[main_idx] = new_node

            elif main_idx in self.branch_post_op_dict.keys():
                pre_node = self.net_dict[main_idx+1]
                new_node = self.net_dict[main_idx]
                new_node = self.get_op(new_node, pre_node)
                self.net_dict[main_idx] = new_node

                pre_node = new_node
                bid_lst = self.branch_post_op_dict[main_idx]
                for i in range(1, len(bid_lst)):
                    bid = bid_lst[i] # find the first op idx of the ith branch
                    branch_op_lst = self.net_lst[self.branchid[bid]] # switch to the branch op lst
                    for j in range(len(branch_op_lst)-1, -1, -1):
                        new_node = self.net_dict[branch_op_lst[j]]
                        new_node = self.get_op(new_node, pre_node)
                        self.net_dict[branch_op_lst[j]] = new_node
                        pre_node = new_node

            elif main_idx in self.merge_dict.keys():
                arg_lst = []
                for a in self.merge_dict[main_idx]:
                    arg_lst.append(self.net_dict[a])
                new_node = self.net_dict[main_idx]
                new_node = self.get_op(new_node, arg_lst=arg_lst)
                self.net_dict[main_idx] = new_node
                
            main_idx -= 1
        return new_node

    def check_consecutive_add(self, node):
        try:
            return node.op.name=='add' and len(node.type_args[1].shape)==3 and node.args[0].op.name=='add' and len(node.args[0].type_args[1].shape)==3
        except:
            return False
    
    def check_branch(self, node):
        try:
            if 'Tuple' not in str(type(node)):
                cnt = 0
                for i in range(len(node.args)):
                    if 'Call' in str(type(node.args[i])):
                        cnt += 1
                return cnt>=2
            else:
                return True
        except:
            return False

    def check_constant(self, node):
        try:
            return 'Constant' in str(type(node))
        except:
            return False

    def check_Var(self, node):
        try:
            return 'Var' in str(type(node))
        except:
            return False

    def get_op(self, node, pre_node=None, arg_lst = None):
        if 'Tuple' in str(type(node)):
            return relay.Tuple(arg_lst)

        if arg_lst is not None:
            args = arg_lst
        else:
            args = []
            for a in node.args:
                if 'Call' in str(type(a)):
                    args.append(pre_node)
                else:
                    args.append(a)

        if node.op.name=='nn.conv2d':
            return relay.nn.conv2d(args[0], args[1], **node.attrs)
        elif node.op.name=='nn.relu':
            return relay.nn.relu(args[0])
        elif node.op.name=='multiply':
            return relay.multiply(args[0], args[1])
        elif node.op.name=='add':
            return relay.add(args[0], args[1])
        elif node.op.name=='concatenate':
            return relay.concatenate(pre_node, **node.attrs)
        elif node.op.name=='nn.max_pool2d':
            return relay.nn.max_pool2d(args[0], **node.attrs)
        elif node.op.name=='nn.avg_pool2d':
            return relay.nn.avg_pool2d(args[0], **node.attrs)
        elif node.op.name=='nn.global_avg_pool2d':
            return relay.nn.global_avg_pool2d(args[0], **node.attrs)
        elif node.op.name=='nn.batch_flatten':
            return relay.nn.batch_flatten(args[0])
        elif node.op.name=='nn.dense':
            return relay.nn.dense(args[0], args[1], **node.attrs)
        elif node.op.name=='nn.dropout':
            return relay.nn.dropout(args[0], **node.attrs)
        else:
            return False

@relay.op.register_alter_op_layout("nn.conv2d", level=114)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    def get_shape(tensor):
        if 'Var' in str(type(tensor)):
            return tensor.type_annotation.concrete_shape
        elif 'Constant' in str(type(tensor)):
            return tensor.data.shape
        elif 'TensorType' in str(type(tensor)):
            return tensor.concrete_shape
        else:
            if "pad" in tensor.op.name:
                return tensor.type_args[0].concrete_shape
            return (-1, -1, -1, -1)
    
    if len(get_shape(data))>4 or len(get_shape(weight))>4 or len(get_shape(out_type))>4:
        return relay.nn.conv2d(data, weight, **attrs)

    N, IC, IH, IW = get_shape(data)
    OC, IC, KH, KW = get_shape(weight)
    N, OC, OH, OW = get_shape(out_type)
    PH_L, PW_L, PH_R, PW_R = attrs.get_int_tuple("padding")
    SH, SW = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")

    res = relay.query_layout.AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW)
    new_attrs = dict(attrs)

    src_df, weight_df, dst_df = res.split(',')

    def trans_data(input_data, is_weight=False):
        dic = data_dic
        res = input_data
        if is_weight:
            dic = weight_dic
                
        for key, value in dic.items():
            if key.upper() in input_data:
                res = res.replace(key.upper(), value, 1)
                res = res.replace(key, value.lower(), 1)
            else:
                res = res.replace(key, value, 1)
        return res

    new_attrs['data_layout'] = trans_data(src_df, is_weight=False)
    new_attrs['kernel_layout'] = trans_data(weight_df, is_weight=True)
    new_attrs['out_layout'] = trans_data(dst_df, is_weight=False)

    return relay.nn.conv2d(data, weight, **new_attrs)

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def benchmark(network, batch_size, profiling=False, check_acc=False, warmup=100, batches=400, dtype="float32", target="llvm"):
    ctx = tvm.cpu()
    input_shape = (batch_size, 3, 224, 224)
    if network=="InceptionV3":
        input_shape = (batch_size, 3, 300, 300)
    
    block = gluoncv.model_zoo.get_model(network_dict[network], pretrained=True)
    mod, params = relay.frontend.from_mxnet(
        block, shape={"data": input_shape}, dtype=dtype
    )

    seq = tvm.transform.Sequential(
        [   
            # tvm.transform.PrintIR(),
            relay.transform.CanonicalizeOps(),
            relay.transform.InferType(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            # tvm.transform.PrintIR(),

            CustomPipeline(),
            relay.transform.FoldConstant(),
            # tvm.transform.PrintIR(),
            
            relay.transform.AlterOpLayout(),
            relay.transform.FoldConstant(),
            # tvm.transform.PrintIR(),

            relay.transform.MergeComposite(pattern_table()),
            # tvm.transform.PrintIR(),
            relay.transform.AnnotateTarget("dnnl"),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(),
            # tvm.transform.PrintIR(),
        ]
    )


    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    with tvm.transform.PassContext(opt_level=3):#, instruments=[PrintIR()]):# 
        json, lib, params = relay.build(seq(mod), target=target, params=params)

    if check_acc:
        img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
        img_name = "cat.png"
        img_path = download_testdata(img_url, img_name, module="data")
        image = Image.open(img_path).resize((input_shape[2], input_shape[3]))
        sample = transform_image(image)
        if batch_size>1:
            sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])

        import tvm.contrib.graph_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
    
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        out = rt_mod.run()
        sample_for_mxnet = mx.ndarray.array(sample)
        mxnet_output = block(sample_for_mxnet)
        tvm_output = rt_mod.get_output(0)
        # print("mxnet_output:{}".format(mxnet_output))
        # print("tvm_output:{}".format(tvm_output))
        print("{} mse:{}".format(network, np.mean((tvm_output.asnumpy()-mxnet_output.asnumpy())**2)))
    elif profiling:
        import datetime
        tic = datetime.datetime.now()
        from tvm.contrib.debugger import debug_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
        sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])#np.ones((batch_size, 3, 224, 224))#
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        total_time_lst = []
        for i in range(batches+warmup):
            tmp = rt_mod.profile()
            gap = tmp.calls[1]["Duration (us)"].microseconds
            #percent = tmp.calls[0]["Percent"].percent
            reorder = tmp.calls[2]["Duration (us)"].microseconds
            #total_time = us * 100 / percent / 1000
            print("{}/{}: gap:{:.4f}, reorder:{:.4f}".format(i, batches+warmup, gap, reorder))
            total_time = gap+reorder
            total_time_lst.append(total_time)
        print("network:{}".format(network))
        print("all ops' execution time:{}".format(np.mean(total_time_lst[warmup::])))
        print("all ops' execution time:{}".format(np.mean(total_time_lst[warmup::])/1000))
        print("profiling time:{}".format(datetime.datetime.now()-tic))
    
    else:
        import tvm.contrib.graph_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)
        sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        for i in range(batches+warmup):
            if i == warmup:
                tic = time.time()
            out = rt_mod.run()
        with_fuse_fps = batches * batch_size / (time.time() - tic)
        print("{}: with_fuse_fps: {:.4f} fps".format(network, with_fuse_fps))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                "vgg11", "vgg13", "vgg16", "vgg19", 
                "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                "densenet121", "InceptionV3", "all"],
        default="all",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--check_acc", type=bool, default=True)
    args = parser.parse_args()

    if args.network == "all":
        networks = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                    "vgg11", "vgg13", "vgg16", "vgg19", 
                    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                    "densenet121", 
                    "InceptionV3"]
    else:
        networks = [args.network]

    target = tvm.target.Target(args.target)

    for network in networks:
        benchmark(network, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
        warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)
