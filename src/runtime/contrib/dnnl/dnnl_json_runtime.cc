/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"
#include <map>
#include <sys/time.h>

//debug
#include <iostream>

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;
class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  std::map<std::string, tag> layout_dict {
    {"NCHW", tag::abcd},//tag::abcd
    {"OIHW", tag::abcd},//tag::abcd
    {"NCHW16c", tag::aBcd16b}, //tag::aBcd16b
    {"NCHW8c", tag::aBcd8b}, //tag::aBcd8b
    {"OIHW16o16i", tag::ABcd16a16b},//OIhw16o16i
    {"OIHW16i16o", tag::ABcd16b16a},
    {"OIHW8o8i", tag::ABcd8a8b},
    {"OIHW8i8o", tag::ABcd8b8a},
    
    {"NHWC", tag::acdb},
    {"OHWI64o", tag::Acdb64a},
    {"OHWI48o", tag::Acdb48a},
    {"OHWI32o", tag::Acdb32a},
    {"OHWI16o", tag::Acdb16a},
    {"OHWI8o", tag::Acdb8a},
    
    {"NC16c64n", tag::AB16b64a},
    {"NC16c32n", tag::AB16b32a},
    {"NC16c16n", tag::AB16b16a},
    };

  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);  
      entry_out_mem_[eid].first.set_data_handle(data_entry_[eid]->data);
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      entry_out_mem_[eid].first.set_data_handle(data_entry_[eid]->data);
    }
    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      net_.at(i).execute(stream_, net_args_.at(i));
    }
    stream_.wait();
  }

 private:
  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("dnnl.conv2d_bias" == op_name) {
          Conv2d(nid, false, true);
        } else if ("dnnl.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("dnnl.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
        } else if ("dnnl.conv2d_bias_sum_relu" == op_name) {
          Conv2d(nid, true, true, true);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("add" == op_name) {
          Binary(nid, dnnl::algorithm::binary_add);
        } else if ("multiply" == op_name) {
          Binary(nid, dnnl::algorithm::binary_mul);
        } else if ("concatenate" == op_name) {
          Concat(nid);
        } else if ("nn.max_pool2d" == op_name) {
          Pooling(nid, dnnl::algorithm::pooling_max);
        } else if ("nn.avg_pool2d" == op_name) {
          Pooling(nid, dnnl::algorithm::pooling_avg);
        } else if ("dnnl.special_matmul_bias_gelu" == op_name) {
          SpecialMatmul(nid, true, "gelu");
        } else if ("dnnl.special_matmul_bias_relu" == op_name) {
          SpecialMatmul(nid, true, "relu");
        } else if ("dnnl.special_matmul_bias_mul" == op_name) {
          SpecialMatmul(nid, true, "none", true);
        } else if ("dnnl.special_matmul_div_add" == op_name) {
          SpecialMatmul(nid, false, "none", false, true, true);
        } else if ("dnnl.special_matmul_bias_mul_add" == op_name) {
          SpecialMatmul(nid, true, "none", true, false, true);
        } else if ("dnnl.special_matmul_bias" == op_name) {
          SpecialMatmul(nid, true, "none");
        } else if ("nn.special_matmul" == op_name) {
          SpecialMatmul(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Bind a JSON graph node entry to a DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
    }
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK_EQ(dltype.bits, 32);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false, const bool has_sum = false) {
    auto node = nodes_[nid];
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    auto src_df = layout_dict[node.GetAttr<std::vector<std::string>>("data_layout")[0]];
    auto weight_df = layout_dict[node.GetAttr<std::vector<std::string>>("kernel_layout")[0]];
    std::vector<std::string> outC = node.GetAttr<std::vector<std::string>>("channels");

    //assume the default layout as NHWC
    dnnl::memory::dim N = input_shape[0],       // batch size
        IH = input_shape[1],                    // input height
        IW = input_shape[2],                    // input width
        IC = input_shape[3],                    // input channels
        OC = std::stoi(outC[0]),                // output channels
        KH = weight_shape[1],                   // weight height
        KW = weight_shape[2],                   // weight width
        PH_L = std::stoi(str_padding[0]),       // height padding: left
        PH_R = std::stoi(str_padding[2]),       // height padding: right
        PW_L = std::stoi(str_padding[1]),       // width padding: left
        PW_R = std::stoi(str_padding[3]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[1]);         // weight-wise stride
    
    if (node.GetAttr<std::vector<std::string>>("data_layout")[0].substr(0,4)=="NCHW") {
        IC = input_shape[1];
        IH = input_shape[2];
        IW = input_shape[3];
        if (input_shape.size()==5) {
          IC = IC * input_shape[input_shape.size()-1];
        }
    }
    if (node.GetAttr<std::vector<std::string>>("kernel_layout")[0].substr(0,4)=="OIHW") {
        KH = weight_shape[2];
        KW = weight_shape[3];
    }

    dnnl::memory::dim OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
                      OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::f32, src_df);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, weight_df);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::x);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::any);

    // Covn2d description.
    auto conv_desc = has_bias? 
        dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r) : \ 

        dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);
    

    // Enable ReLU
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;
    if (has_sum) {
      ops.append_sum(1.f);
      attr.set_post_ops(ops);
    }
    if (has_relu) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 1.f);
      attr.set_post_ops(ops);
    }

    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    auto conv2d_src_memory = BindDNNLMemory(data_entry, conv_src_md);

    // Weight memory.
    auto conv2d_weights_memory = BindDNNLMemory(
        weight_entry, conv_weights_md);

    // Bias memory.
    auto conv2d_bias_memory = dnnl::memory(conv_bias_md, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, conv2d_bias_memory);
    }

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = dnnl::memory(conv2d_prim_desc.dst_desc(), engine_);
    if (has_sum) {
      auto dst_entry = node.GetInputs()[3];
      conv2d_dst_memory = BindDNNLMemory(dst_entry, conv2d_prim_desc.dst_desc());
    }
      BindDNNLMemory(out_entry, conv2d_dst_memory);

    // Bind memory buffers.
    if (has_bias) {
      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_BIAS, conv2d_bias_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
    } 
    else {
      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
    }
    
  }

  void SpecialMatmul(const size_t& nid, const bool has_bias = false, const std::string act_type = "none",
             const bool has_mul = false, const bool has_div = false, const bool has_add = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    auto weight_df = layout_dict[node.GetAttr<std::vector<std::string>>("weight_layout")[0]];
    bool is_batch_matmul = false;
    if (input_shape.size() == 4 && weight_shape.size() == 4) {
        is_batch_matmul = true;
    }

    if (is_batch_matmul) {
        dnnl::memory::dim B1 = input_shape[0],
            B2 = input_shape[1],
            M = input_shape[2],
            K = input_shape[3],
            N = weight_shape[3];

        // Memory_shapes.
        dnnl::memory::dims data_dims = {B1, B2, M, K};
        dnnl::memory::dims weight_dims = {B1, B2, K, N};
        dnnl::memory::dims bias_dims = {B1, B2, M, N};
        dnnl::memory::dims out_dims = {B1, B2, M, N};

        // Memory descriptions.
        auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nchw});
        auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, weight_df});
        auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::nchw});
        auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nchw});

        auto matmul_desc = has_bias?
            dnnl::matmul::desc(data_md, weight_md, bias_md, dst_md):
            dnnl::matmul::desc(data_md, weight_md, dst_md);
        
        dnnl::post_ops ops;
        if (act_type == "relu") {
            ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 1.f);
        }
        if (act_type == "gelu") {
            ops.append_eltwise(1.f, dnnl::algorithm::eltwise_gelu, 0.f, 1.f);
        }
        if (has_mul == true) {
            ops.append_binary(dnnl::algorithm::binary_mul, dst_md);
        }
        if (has_div == true) {
            ops.append_binary(dnnl::algorithm::binary_div, dst_md);
        }
        if (has_add == true) {
            ops.append_binary(dnnl::algorithm::binary_add, dst_md);
        }
        dnnl::primitive_attr attr;
        attr.set_post_ops(ops);
        
        auto matmul_prim_desc = dnnl::matmul::primitive_desc(matmul_desc, attr, engine_);
        auto matmul = dnnl::matmul(matmul_prim_desc);
        net_.push_back(matmul);

        // Memories.
        auto data_memory = BindDNNLMemory(data_entry, data_md);
        auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
        auto bias_memory = dnnl::memory(bias_md, engine_);
        auto mul_div_memory = dnnl::memory(dst_md, engine_);
        auto add_memory = dnnl::memory(dst_md, engine_);
        auto dst_memory = dnnl::memory(dst_md, engine_);
        JSONGraphNodeEntry out_entry(nid, 0);
        
        int entry_count = 2;
        if (has_bias) {
            auto bias_entry = node.GetInputs()[entry_count];
            BindDNNLMemory(bias_entry, bias_memory);
            entry_count++;
        } 
        if (has_mul || has_div) {
            auto mul_div_entry = node.GetInputs()[entry_count];
            BindDNNLMemory(mul_div_entry, mul_div_memory);
            entry_count++;
        }
        if (has_add) {
            auto add_entry = node.GetInputs()[entry_count];
            BindDNNLMemory(add_entry, add_memory);
            entry_count++;
        }
        BindDNNLMemory(out_entry, dst_memory);

        if (has_bias) {
            if (has_mul || has_div) {
                if (has_add) {
                    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                        {DNNL_ARG_WEIGHTS, weight_memory},
                                        {DNNL_ARG_BIAS, bias_memory},
                                        {DNNL_ARG_DST, dst_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, mul_div_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, add_memory}});
                } else {
                    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                        {DNNL_ARG_WEIGHTS, weight_memory},
                                        {DNNL_ARG_BIAS, bias_memory},
                                        {DNNL_ARG_DST, dst_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, add_memory}});
                }
            } else {
                net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                    {DNNL_ARG_WEIGHTS, weight_memory},
                                    {DNNL_ARG_BIAS, bias_memory},
                                    {DNNL_ARG_DST, dst_memory}});
            }
        } else {
            if (has_mul || has_div) {
                if (has_add) {
                    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                        {DNNL_ARG_WEIGHTS, weight_memory},
                                        {DNNL_ARG_DST, dst_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, mul_div_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, add_memory}});
                } else {
                    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                        {DNNL_ARG_WEIGHTS, weight_memory},
                                        {DNNL_ARG_DST, dst_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, mul_div_memory}});
                }
            } else {
                net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                    {DNNL_ARG_WEIGHTS, weight_memory},
                                    {DNNL_ARG_DST, dst_memory}});
            }
        }
    } else {
        dnnl::memory::dim B = input_shape[0],  // batch size
            IC = input_shape[1],               // input channels
            OC = weight_shape[0];              // output channels

        if (input_shape.size() == 3) {
            B = input_shape[0] * input_shape[1];
            IC = input_shape[2];
        }

        auto layout = node.GetAttr<std::vector<std::string>>("weight_layout")[0];
        if ("NC16c64n" == layout || "NC16c32n" == layout || "NC16c16n" == layout) {
            OC = weight_shape[0] * weight_shape[3];
        }

        // std::cout << "B: " << B  << " " << "OC: " << OC << " " << "IC: " << IC << std::endl;
        // Memory shapes.
        dnnl::memory::dims data_dims = {B, IC};
        dnnl::memory::dims weight_dims = {OC, IC};
        dnnl::memory::dims bias_dims = {OC};
        dnnl::memory::dims out_dims = {B, OC};

        // Memory descriptions.
        auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
        auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, weight_df});
        auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
        auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

        // Dense description.
        auto dense_desc = has_bias?
            dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                            weight_md, bias_md, dst_md) : \
            dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                            weight_md, dst_md);
        // std::cout << "[LOG] description created " << std::endl;

        // Enable ReLU
        dnnl::post_ops ops;
        if (act_type == "relu") {
            ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 1.f);
        }
        if (act_type == "gelu") {
            ops.append_eltwise(1.f, dnnl::algorithm::eltwise_gelu_erf, 0.f, 1.f);
        }
        if (has_mul == true) {
            ops.append_binary(dnnl::algorithm::binary_mul, dst_md);
        }
        if (has_add == true) {
            ops.append_binary(dnnl::algorithm::binary_add, dst_md);
        }
        dnnl::primitive_attr attr;
        attr.set_post_ops(ops);

        auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, attr, engine_);
        // std::cout << "[LOG] PD created " << std::endl;
        
        auto dense = dnnl::inner_product_forward(dense_prim_desc);
        net_.push_back(dense);

        // std::cout << "[LOG] primitive created " << std::endl;

        // Memories.
        auto data_memory = BindDNNLMemory(data_entry, data_md);
        auto weight_memory = BindDNNLMemory(weight_entry, dense_prim_desc.weights_desc());
        auto bias_memory = dnnl::memory(bias_md, engine_);
        // std::cout << "dense_prim_desc.dst_desc().dims().size(): " << dense_prim_desc.dst_desc().dims().size() << std::endl;
        // std::cout << dense_prim_desc.dst_desc().dims()[0] << " " << dense_prim_desc.dst_desc().dims()[1] << std::endl;
        auto mul_memory = dnnl::memory(dense_prim_desc.dst_desc(), engine_);
        auto add_memory = dnnl::memory(dense_prim_desc.dst_desc(), engine_);
        auto dst_memory = dnnl::memory(dense_prim_desc.dst_desc(), engine_);

        // std::cout << "[LOG] memory created" << std::endl;

        JSONGraphNodeEntry out_entry(nid, 0);

        int entry_count = 2;
        if (has_bias) {
            auto bias_entry = node.GetInputs()[entry_count];
            BindDNNLMemory(bias_entry, bias_memory);
            entry_count++;
        } 
        if (has_mul) {
            auto mul_entry = node.GetInputs()[entry_count];
            BindDNNLMemory(mul_entry, mul_memory);
            entry_count++;
        }
        if (has_add) {
            auto add_entry = node.GetInputs()[entry_count];
            BindDNNLMemory(add_entry, add_memory);
            entry_count++;
        }
        BindDNNLMemory(out_entry, dst_memory);

        if (has_bias) {
            if (has_mul) {
                if (has_add) {
                    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                        {DNNL_ARG_WEIGHTS, weight_memory},
                                        {DNNL_ARG_BIAS, bias_memory},
                                        {DNNL_ARG_DST, dst_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, mul_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, add_memory}});
                } else {
                    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                        {DNNL_ARG_WEIGHTS, weight_memory},
                                        {DNNL_ARG_BIAS, bias_memory},
                                        {DNNL_ARG_DST, dst_memory},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, mul_memory}});
                }
            } else {
                net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                    {DNNL_ARG_WEIGHTS, weight_memory},
                                    {DNNL_ARG_BIAS, bias_memory},
                                    {DNNL_ARG_DST, dst_memory}});
            }
        } else {
            net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                                {DNNL_ARG_WEIGHTS, weight_memory},
                                {DNNL_ARG_DST, dst_memory}});
        }
    }
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim B = input_shape[0],  // batch size
        IC = input_shape[1],               // input channels
        OC = weight_shape[0];              // output channels

    // Memory shapes.
    dnnl::memory::dims data_dims = {B, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    auto bias_memory = dnnl::memory(bias_md, engine_);
    float bias[OC] = {0};
    write_to_dnnl_memory(bias, bias_memory, OC * sizeof(float));
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    dnnl::memory::dims data_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim IC = data_shape[1];
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    // Memory description.
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

    // BN description.
    auto bn_desc = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, data_md, epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(bn_desc, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_prim_desc);
    net_.push_back(bn);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);
    auto mean_memory = BindDNNLMemory(mean_entry, bn_prim_desc.mean_desc());
    auto variance_memory = BindDNNLMemory(variance_entry, bn_prim_desc.variance_desc());

    // In DNNL, weight is composed of gamma+beta, so we point them to the same DNNL memory but
    // assign an offset to beta data for runtime serialization.
    auto weight_memory = BindDNNLMemory(gamma_entry, bn_prim_desc.weights_desc(), 0);
    BindDNNLMemory(beta_entry, weight_memory, IC);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_DST, out_memory},
                         {DNNL_ARG_SCALE_SHIFT, weight_memory},
                         {DNNL_ARG_MEAN, mean_memory},
                         {DNNL_ARG_VARIANCE, variance_memory}});
  }

  void Relu(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(shape, dt::f32);

    auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::eltwise_relu, data_md, 0);
    auto relu_prim_desc = dnnl::eltwise_forward::primitive_desc(relu_desc, engine_);
    ICHECK(data_md == relu_prim_desc.dst_desc());

    auto relu = dnnl::eltwise_forward(relu_prim_desc);
    net_.push_back(relu);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Binary(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];

    // Memory and compute description.
    std::vector<dnnl::memory::dims> data_dims;
    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;

    ICHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));
    }
    ICHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    auto binary_desc = dnnl::binary::desc(algo, data_mds[0], data_mds[1], out_md);
    auto binary_prim_desc = dnnl::binary::primitive_desc(binary_desc, engine_);
    auto binary = dnnl::binary(binary_prim_desc);
    net_.push_back(binary);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

  void Concat(const size_t& nid) {
    auto node = nodes_[nid];

    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;
    
    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]); // axis
    
    if (axis < 0) {
        ICHECK_LE(0 - axis, shape.size());
        axis = shape.size() + axis;
    }
    else ICHECK_LT(axis, shape.size());
    
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);
        data_mds.push_back(data_md);
        data_memories.push_back(BindDNNLMemory(entry, data_md));
      }
    
    auto concat_prim_desc = dnnl::concat::primitive_desc(axis, data_mds, engine_);
    auto concat = dnnl::concat(concat_prim_desc);
    net_.push_back(concat);
    
    auto out_md = concat_prim_desc.dst_desc();
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    std::unordered_map<int, dnnl::memory> args;
    for (int i = 0; i < (int)data_memories.size(); i++) {
        args.insert({DNNL_ARG_MULTIPLE_SRC + i, data_memories[i]});
    }
    args.insert({DNNL_ARG_DST, out_memory});
    net_args_.push_back(args);
  }
  
  void Pooling(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim pool_size0 = std::stoi(node.GetAttr<std::vector<std::string>>("pool_size")[0]);
    dnnl::memory::dim pool_size1 = std::stoi(node.GetAttr<std::vector<std::string>>("pool_size")[1]);
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> ceil_mode = node.GetAttr<std::vector<std::string>>("ceil_mode");
    dnnl::memory::dim dilation0 = std::stoi(node.GetAttr<std::vector<std::string>>("dilation")[0]);
    dnnl::memory::dim dilation1 = std::stoi(node.GetAttr<std::vector<std::string>>("dilation")[1]);
    auto src_df = layout_dict[node.GetAttr<std::vector<std::string>>("layout")[0]];

    // Attributes related to AvgPool
    if (algo==dnnl::algorithm::pooling_avg) {
        int int_countpad = std::stoi(node.GetAttr<std::vector<std::string>>("count_include_pad")[0]); //note
        bool count_include_pad = int_countpad != 0 ? true : false;
        algo = count_include_pad ?
          dnnl::algorithm::pooling_avg_include_padding :
          dnnl::algorithm::pooling_avg_exclude_padding ;
    }
    
    //assume the default layout as NHWC
    dnnl::memory::dim N = input_shape[0],
      IH = input_shape[1],
      IW = input_shape[2],
      IC = input_shape[3],
      KH = pool_size0,
      KW = pool_size1,
      PW_L = std::stoi(str_padding[1]),
      PW_R = std::stoi(str_padding[3]),
      PH_L = std::stoi(str_padding[0]),
      PH_R = std::stoi(str_padding[2]),
      SH = std::stoi(str_strides[0]),
      SW = std::stoi(str_strides[1]),
      DH = dilation0,
      DW = dilation1;
      
    if (node.GetAttr<std::vector<std::string>>("layout")[0].substr(0,4)=="NCHW") {
        IC = input_shape[1];
        IH = input_shape[2];
        IW = input_shape[3];
        if (input_shape.size()==5) {
          IC = IC * input_shape[input_shape.size()-1];
        }
    }

    dnnl::memory::dim OH = (IH - KH + PH_L + PH_R) / SH + 1,
                      OW = (IW - KW + PW_L + PW_R) / SW + 1;
    
    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims kernel_dims = {KH, KW}; // modified
    dnnl::memory::dims dst_dims = {N, IC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};
    dnnl::memory::dims dilation = {DH, DW};

    // Memory descriptions.
    auto pool_src_md = dnnl::memory::desc(src_dims, dt::f32, src_df);
    auto pool_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::any);

    // Pooling description.
    auto pool_desc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, algo,
      pool_src_md, pool_dst_md, strides_dims, kernel_dims,
      padding_dims_l, padding_dims_r
    );
    auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(pool_desc, engine_, true);
    auto pool = dnnl::pooling_forward(pool_prim_desc);
    net_.push_back(pool);

    // Memories.
    auto pool2d_src_memory = BindDNNLMemory(data_entry, pool_src_md);

    JSONGraphNodeEntry out_entry(nid, 0);
    auto pool2d_dst_memory = BindDNNLMemory(out_entry, pool_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({
      {DNNL_ARG_SRC, pool2d_src_memory},
      {DNNL_ARG_DST, pool2d_dst_memory}
    });
  }

//   void Matmul(const size_t& nid, const bool has_bias = false, const std::string act_type = "none",
//     const bool has_mul = false, const bool has_add = false) {
//     auto node = nodes_[nid];

//     // Setup attributes.
//     auto data_entry = node.GetInputs()[0];
//     auto weight_entry = node.GetInputs()[1];

//     dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
//     dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

//     dnnl::memory::dim M = input_shape[0],  // batch size
//         K = input_shape[1],               // input channels
//         N = weight_shape[1];              // output channels

//     // Memory shapes.
//     dnnl::memory::dims data_dims = {M, K};
//     dnnl::memory::dims weight_dims = {K, N};
//     dnnl::memory::dims bias_dims = {M, N};
//     dnnl::memory::dims out_dims = {M, N};

//     // Memory descriptions.
//     auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::ab});
//     auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::ab});
//     auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::ab});
//     auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::ab});

//     auto matmul_desc = has_bias?
//         dnnl::matmul::desc(data_md, weight_md, bias_md, dst_md):
//         dnnl::matmul::desc(data_md, weight_md, dst_md);
    
//     dnnl::post_ops ops;
//     if (act_type == "relu") {
//       ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 1.f);
//     }
//     if (act_type == "gelu") {
//       ops.append_eltwise(1.f, dnnl::algorithm::eltwise_gelu, 0.f, 1.f);
//     }
//     if (has_mul == true) {
//       ops.append_binary(dnnl::algorithm::binary_mul, dst_md);
//     }
//     if (has_add == true) {
//       ops.append_binary(dnnl::algorithm::binary_add, dst_md);
//     }
//     dnnl::primitive_attr attr;
//     attr.set_post_ops(ops);
    
//     auto matmul_prim_desc = dnnl::matmul::primitive_desc(matmul_desc, attr, engine_);
//     auto matmul = dnnl::matmul(matmul_prim_desc);
//     net_.push_back(matmul);

//     // Memories.
//     auto data_memory = BindDNNLMemory(data_entry, data_md);
//     auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
//     auto bias_memory = dnnl::memory(bias_md, engine_);
//     auto mul_memory = dnnl::memory(matmul_prim_desc.dst_desc(), engine_);
//     auto add_memory = dnnl::memory(matmul_prim_desc.dst_desc(), engine_);
//     auto dst_memory = dnnl::memory(matmul_prim_desc.dst_desc(), engine_);
//     JSONGraphNodeEntry out_entry(nid, 0);
//     if (has_bias) {
//       auto bias_entry = node.GetInputs()[2];
//       BindDNNLMemory(bias_entry, bias_memory);
//     } 
//     if (has_mul) {
//       auto mul_entry = node.GetInputs()[3];
//       dst_memory = BindDNNLMemory(mul_entry, mul_memory);
//     }
//     if (has_add) {
//       auto add_entry = node.GetInputs()[4];
//       dst_memory = BindDNNLMemory(add_entry, add_memory);
//     }
//     BindDNNLMemory(out_entry, dst_memory);

//     if (has_bias) {
//         if (has_mul) {
//             if (has_add) {
//                 net_args_.push_back({{DNNL_ARG_SRC, data_memory},
//                                     {DNNL_ARG_WEIGHTS, weight_memory},
//                                     {DNNL_ARG_BIAS, bias_memory},
//                                     {DNNL_ARG_DST, dst_memory},
//                                     {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, mul_memory},
//                                     {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, dst_memory}});
//             } else {
//                 net_args_.push_back({{DNNL_ARG_SRC, data_memory},
//                                     {DNNL_ARG_WEIGHTS, weight_memory},
//                                     {DNNL_ARG_BIAS, bias_memory},
//                                     {DNNL_ARG_DST, dst_memory},
//                                     {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, dst_memory}});
//             }
//         } else {
//             net_args_.push_back({{DNNL_ARG_SRC, data_memory},
//                                 {DNNL_ARG_WEIGHTS, weight_memory},
//                                 {DNNL_ARG_BIAS, bias_memory},
//                                 {DNNL_ARG_DST, dst_memory}});
//         }
//         } else {
//         net_args_.push_back({{DNNL_ARG_SRC, data_memory},
//                             {DNNL_ARG_WEIGHTS, weight_memory},
//                             {DNNL_ARG_DST, dst_memory}});
//     }
//   }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
  }

  // Generate DNNL memory description and infer the data layout by the given shape.
  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
    dnnl::memory::desc data_md;
    switch (shape.size()) {
      case 1:
        data_md = dnnl::memory::desc({shape, dtype, tag::a});
        break;
      case 2:
        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::pair<dnnl::memory, size_t>> entry_out_mem_;
};

runtime::Module DNNLJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
