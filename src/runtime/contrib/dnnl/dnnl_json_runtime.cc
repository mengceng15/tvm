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
#include <regex>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    // BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);

    BuildEngine();
  }

  void Run() override {
    // Fill in the input buffers.
    std::cout << "data_entry_ size: " << data_entry_.size() << std::endl;
    std::cout << "input_nodes_ size: " << input_nodes_.size() << std::endl;
    std::cout << "outputs_ size: " << outputs_.size() << std::endl;
    std::cout << "entry_out_mem_ size: " << entry_out_mem_.size() << std::endl;

    if (data_entry_.size() != entry_out_mem_.size()) {
      std::cout << "data_entry_ and entry_out_mem_ size mismatch" << std::endl;
      std::cout << data_entry_.size() << " vs " << entry_out_mem_.size() << std::endl;
    }

    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      std::cout << "current eid: " << eid << std::endl;
      // TODO(@comaniac): Support other data lengths.
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);

      {
        std::cout << "shape of input[" << i << "]:" << std::endl;
        int n_dims = data_entry_[eid]->ndim;
        std::cout << "n_dims: " << n_dims << std::endl;
        int64_t* ptr = data_entry_[eid]->shape;
        int size = 1;
        for (int d = 0; d < n_dims; d++) {
          std::cout << *(ptr + d) << " ";
          size *= *(ptr + d);
        }

        for (int i = 0; i < size; i++) {
          std::cout << *((float*) data_entry_[eid]->data + i) << " ";
        }
        std::cout << std::endl;
        
        // if (n_dims == 0) {
        //   std::cout << *ptr << std::endl;
        // }
        std::cout << std::endl;
      }

      write_to_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                           offset_in_bytes);
    }
    std::cout << "data loading success" << std::endl;

    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      net_.at(i).execute(stream_, net_args_.at(i));
      std::cout << "finish running net " << i << std::endl;
    }
    stream_.wait();

    // Read output buffers.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      read_from_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                            offset_in_bytes);
    }
    std::cout << "data writing success " << std::endl;
  }

 private:
  // Build up the engine based on the input graph.

  std::map<std::string, dnnl::algorithm> elt_name2algo{
      {"abs", dnnl::algorithm::eltwise_abs},
      {"exp", dnnl::algorithm::eltwise_exp},
      {"log", dnnl::algorithm::eltwise_log},
      {"sqrt", dnnl::algorithm::eltwise_sqrt},
      {"round", dnnl::algorithm::eltwise_round},
      {"logsumexp", dnnl::algorithm::eltwise_logsigmoid},
      {"nn.relu", dnnl::algorithm::eltwise_relu},
      {"nn.leaky_relu", dnnl::algorithm::eltwise_relu},
      {"tanh", dnnl::algorithm::eltwise_tanh},
      {"sigmoid", dnnl::algorithm::eltwise_logistic},
      {"clip", dnnl::algorithm::eltwise_clip},
  };

  std::map<std::string, tag> layout_dict{
      {"", tag::any},
      {"NCW", tag::ncw},
      {"NWC", tag::nwc},
      {"OIW", tag::oiw},
      {"GOIW", tag::goiw},
      {"NCHW", tag::nchw},
      {"NHWC", tag::nhwc},
      {"OIHW", tag::oihw},
      {"GOIHW", tag::goihw},
      {"NCDHW", tag::ncdhw},
      {"NDHWC", tag::ndhwc},
      {"OIDHW", tag::oidhw},
      {"GOIDHW", tag::goidhw},
      {"IOHW", tag::iohw},
      {"GIOHW", tag::giohw},
      {"IODHW", tag::iodhw},
      {"GIODHW", tag::giodhw},

      // Blocking layout.
      {"NCW8c", tag::nCw8c},
      {"NCW16c", tag::nCw16c},
      {"OIW16i16o", tag::OIw8i8o},
      {"OIW16i16o", tag::OIw16i16o},
      {"OWI8o", tag::Owi8o},
      {"OWI16o", tag::Owi16o},
      {"NCHW4c", tag::nChw4c},
      {"NCHW8c", tag::nChw8c},
      {"NCHW16c", tag::nChw16c},
      {"OIHW8i8o", tag::OIhw8i8o},
      {"IOHW8i8o", tag::any},
      {"OIHW16i16o", tag::OIhw16i16o},
      {"IOHW16i16o", tag::IOhw16i16o},
      {"GOIHW4i4o", tag::gOIhw4i4o},
      {"GOIHW8i8o", tag::gOIhw8i8o},
      {"GOIHW16i16o", tag::gOIhw16i16o},
      {"OHWI8o", tag::Ohwi8o},
      {"OHWI16o", tag::Ohwi16o},
      {"OHWI32o", tag::Ohwi32o},
      {"OHWI48o", tag::Ohwi48o},
      {"OHWI64o", tag::Ohwi64o},
      {"GOIHW8g", tag::Goihw8g},
      {"GOIHW16g", tag::Goihw16g},
      {"NCDHW8c", tag::nCdhw8c},
      {"NCDHW16c", tag::nCdhw16c},
      {"OIDHW16i16o", tag::OIdhw16i16o},
      {"IODHW16i16o", tag::IOdhw16i16o},
      {"OIDHW8i8o", tag::OIdhw8i8o},
      {"IODHW8i8o", tag::any},
      {"ODHWI8o", tag::Odhwi8o},
      {"ODHWI16o", tag::Odhwi16o},
      {"ODHWI32o", tag::Odhwi32o},
      {"ODHWI48o", tag::Odhwi48o},
      {"ODHWI64o", tag::Odhwi64o},
  };

  bool ParsingOpName(const std::string op_name, dnnl::primitive_attr attr) {
    // Define RegExp.
    std::regex bias_add_pat(".*_bias.*");
    std::regex relu_pat(".*_relu.*");
    std::regex tanh_pat(".*_tanh.*");
    std::regex sigmoid_pat(".*_sigmoid.*");

    // Parsing post-ops.
    dnnl::post_ops ops;
    if (std::regex_match(op_name, relu_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    }
    if (std::regex_match(op_name, tanh_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
    }
    if (std::regex_match(op_name, sigmoid_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
    }
    attr.set_post_ops(ops);

    // Parsing bias_add.
    return std::regex_match(op_name, bias_add_pat) ? true : false;
  }

  dnnl::memory::dims TransDims2Plain(dnnl::memory::dims input_dims, std::string layout) {
    std::vector<char> axis = {
        'N', 'C', 'O', 'I', 'D', 'H', 'W',
    };
    dnnl::memory::dims out_dims;
    std::string::iterator t = layout.begin();
    // Remove numbers in layout string to match the size of input_dims
    while (t != layout.end()) {
      if (*t >= '0' && *t <= '9') {
        layout.erase(t);
      } else {
        t++;
      }
    }
    // Push the correct shapes of each axis into the output_dims
    for (auto a : axis) {
      dnnl::memory::dim shape = 1;
      if (layout.find(a) != std::string::npos) {
        shape *= input_dims[layout.find(a)];
        char lower_a = std::tolower(a);
        if (layout.find(lower_a) != std::string::npos) {
          shape *= input_dims[layout.find(lower_a)];
        }
        out_dims.push_back(shape);
      }
    }
    // Multiply O and I with G, respectively
    if (layout.find("G") != std::string::npos) {
      dnnl::memory::dim G = 1;
      if (layout.find("g") != std::string::npos) {
        G = input_dims[layout.find("g")] * input_dims[layout.find("G")];
      } else {
        G = input_dims[layout.find("G")];
      }
      out_dims[0] *= G;
      out_dims[1] *= G;
    }
    return out_dims;
  }

  dnnl::memory::dims TransformStr2Dims(std::vector<std::string> strs, bool dilates = false) {
    dnnl::memory::dims out_dims;
    if (dilates) {
      std::transform(strs.begin(), strs.end(), std::back_inserter(out_dims),
                     [](const std::string& str) { return std::stoi(str) - 1; });
    } else {
      std::transform(strs.begin(), strs.end(), std::back_inserter(out_dims),
                     [](const std::string& str) { return std::stoi(str); });
    }
    return out_dims;
  }

  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    std::regex conv_pat(".*conv[1-3]d.*");
    std::regex deconv_pat(".*deconv[1-3]d.*");
    std::regex conv_transpose_pat(".*conv[1-3]d_transpose.*");
    std::regex dense_pat(".*dense.*");
    std::regex max_pool_pat(".*max_pool[1-3]d");
    std::regex avg_pool_pat(".*avg_pool[1-3]d");

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        std::cout << "OP_NAME(runtime): " << op_name << std::endl;
        if ("dnnl.qint8_dense_relu" == op_name) {
          Qint8_Dense_RELU(nid);
        } else if ("dnnl.qint8_conv2d" == op_name) {
          Qint8_Conv2d(nid);
        } else if (std::regex_match(op_name, deconv_pat) ||
            std::regex_match(op_name, conv_transpose_pat)) {
          Deconvolution(nid);
        } else if (std::regex_match(op_name, conv_pat)) {
          Convolution(nid);
        } else if (std::regex_match(op_name, dense_pat)) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if (std::regex_match(op_name, max_pool_pat)) {
          Pooling(nid, dnnl::algorithm::pooling_max);
        } else if (std::regex_match(op_name, avg_pool_pat)) {
          Pooling(nid, dnnl::algorithm::pooling_avg);
        } else if (elt_name2algo.count(op_name)) {
          Eltwise(nid);
        } else if ("nn.softmax" == op_name) {
          Softmax(nid);
        } else if ("add" == op_name) {
          Binary(nid, dnnl::algorithm::binary_add);
        } else if ("multiply" == op_name) {
          Binary(nid, dnnl::algorithm::binary_mul);
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
    // Since the DNNL memory has been created before calling this function, we assume the entry
    // has not yet been bound to the other DNNL memory; otherwise it may have memory leak.
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK_EQ(dltype.bits, 32);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  uint32_t GetDataCount(const DLTensor& arr) {
    size_t count = 1;
    for (tvm_index_t i = 0; i < arr.ndim; ++i) {
      count *= static_cast<size_t>(arr.shape[i]);
    }
    return count;
  }

  void Qint8_Dense_RELU(const size_t& nid) {
    std::cout << "Qint8_Dense_RELU" << std::endl;
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    std::cout << "Node number of inputs: " << node.GetInputs().size() << std::endl;
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    // auto bias_entry = node.GetInputs()[2];

    // auto data_scale_entry = node.GetInputs()[2];
    // auto data_min_entry = node.GetInputs()[3];
    // auto data_max_entry = node.GetInputs()[4];
    // auto weight_scale_entry = node.GetInputs()[5];
    // auto weight_min_entry = node.GetInputs()[6];
    // auto weight_max_entry = node.GetInputs()[7];

    JSONGraphNodeEntry out_entry(nid, 0);

#define GET_NODE_INFO(n) \
  int const##n##_json_node_entry_id = EntryID(node.GetInputs()[n].id_, 0); \
  std::cout << node.GetInputs()[n].id_ << std::endl;\
  std::cout << node.GetInputs()[n].index_ << std::endl;\
  std::cout << "const node json node entry id: " << const##n##_json_node_entry_id << std::endl; \
  auto const##n##_tensor_ptr = data_entry_[const##n##_json_node_entry_id]; \
  auto const##n##_ndim = const##n##_tensor_ptr->ndim; \
  std::cout << "data count: " << this->GetDataCount(*const##n##_tensor_ptr) << std::endl; \
  std::cout << "First elemt of const node: " << *((float*)const##n##_tensor_ptr->data) << '\n' << std::endl;

    // GET_NODE_INFO(0) // not const cannot get
    GET_NODE_INFO(1)
    GET_NODE_INFO(2)
    GET_NODE_INFO(3)
    GET_NODE_INFO(4)
    GET_NODE_INFO(5)
    GET_NODE_INFO(6)
    // GET_NODE_INFO(7)

    std::cout << "nodes_ size: " << nodes_.size() << std::endl;
    std::cout << "node input size: " << node.GetInputs().size() << std::endl;
    // nodes_ is global, node is local
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];

    int64_t M = input_shape[0];
    int64_t K = input_shape[1];
    int64_t N = weight_shape[0];

    /*
    std::cout << "M: " << M << " " << "K: " << K << " " << "N: " << N << std::endl;

    std::cout << "input_shape: ";
    for (auto e : input_shape) {
      std::cout << e << " ";
    }
    std::cout << std::endl;
    std::cout << "weight_shape: ";
    for (auto e : weight_shape) {
      std::cout << e << " ";
    }
    std::cout << std::endl;
    */

    std::vector<float> data_scales;
    std::vector<float> weight_scales;
    std::vector<float> bias_scales = {1.0f};
    std::vector<float> dst_scales; // What is this?
    std::vector<float> ip_scales = {1.0f};

    for (uint32_t i = 0; i < this->GetDataCount(*const1_tensor_ptr); i++) {
      float data_scale = *((float*)const1_tensor_ptr->data + i);
      float weight_scale = *((float*)const4_tensor_ptr->data + i);
      data_scales.push_back(1.0f / data_scale);
      weight_scales.push_back(1.0f / weight_scale);
      dst_scales.push_back(data_scale * weight_scale);
    }

    std::cout << "data_scales: " << std::endl;
    for (auto e : data_scales) {
      std::cout << e << " ";
    }
    std::cout << std::endl;

    std::cout << "weight_scales: " << std::endl;
    for (auto e : weight_scales) {
      std::cout << e << " ";
    }
    std::cout << std::endl;

    std::cout << "dst_scales: " << std::endl;
    for (auto e : dst_scales) {
      std::cout << e << " ";
    }
    std::cout << std::endl;

    dnnl::memory::dims ip_src_tz = {M, K};
    dnnl::memory::dims ip_weights_tz = {N, K};
    dnnl::memory::dims ip_bias_tz = {N};
    dnnl::memory::dims ip_dst_tz = {M, N};

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;
    const int dst_mask = 0;
    const int ip_mask = 0;

    auto user_src_md = dnnl::memory::desc({ip_src_tz, dt::f32, tag::ab});
    auto user_wei_md = dnnl::memory::desc({ip_weights_tz, dt::f32, tag::ab});
    auto user_bia_md = dnnl::memory::desc({ip_bias_tz, dt::f32, tag::x});

    // auto user_src_memory = BindDNNLMemory(data_entry, user_src_md);
    // auto user_wei_memory = BindDNNLMemory(weight_entry, user_wei_md);
    // auto user_bia_memory = BindDNNLMemory(bias_entry, user_bia_md);
    auto user_src_memory = BindDNNLMemory(node.GetInputs()[0], user_src_md);
    auto user_wei_memory = BindDNNLMemory(node.GetInputs()[7], user_wei_md);
    auto user_bia_memory = dnnl::memory(user_bia_md, engine_);
    float bias[N] = {0};
    write_to_dnnl_memory(bias, user_bia_memory, N * sizeof(float));

    // debug
    dnnl::memory::dims ip_const_tz = {1};
    auto user_const_md = dnnl::memory::desc({ip_const_tz, dt::f32, tag::x});
    auto user_data_scale_memory = BindDNNLMemory(node.GetInputs()[1], user_const_md);
    auto user_data_min_memory = BindDNNLMemory(node.GetInputs()[2], user_const_md);
    auto user_data_max_memory = BindDNNLMemory(node.GetInputs()[3], user_const_md);
    auto user_weight_scale_memory = BindDNNLMemory(node.GetInputs()[4], user_const_md);
    auto user_weight_min_memory = BindDNNLMemory(node.GetInputs()[5], user_const_md);
    auto user_weight_max_memory = BindDNNLMemory(node.GetInputs()[6], user_const_md);
    // debug end
    
    auto ip_src_md = dnnl::memory::desc({ip_src_tz}, dt::u8, tag::any);
    auto ip_bias_md = dnnl::memory::desc({ip_bias_tz}, dt::s8, tag::any);
    auto ip_weights_md = dnnl::memory::desc({ip_weights_tz}, dt::s8, tag::any);
    // auto ip_dst_md = dnnl::memory::desc({ip_dst_tz}, dt::u8, tag::any);
    auto ip_dst_md = dnnl::memory::desc({ip_dst_tz}, dt::s32, tag::any);

    auto ip_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, ip_src_md,
                    ip_weights_md, ip_bias_md, ip_dst_md);
    dnnl::primitive_attr ip_attr;
    ip_attr.set_output_scales(ip_mask, ip_scales);
    
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    dnnl::post_ops ops;
    ops.append_eltwise(ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
    ip_attr.set_post_ops(ops);

    /*
    try {
      auto ip_pd = dnnl::inner_product_forward::primitive_desc(
          ip_desc, ip_attr, engine_);
    } catch (error &e) {
      if (e.status == dnnl_unimplemented)
          throw example_allows_unimplemented {
                  "No int8 ip implementation is available for this "
                  "platform.\n"
                  "Please refer to the developer guide for details."};

      // on any other error just re-throw
      throw;
    }
    */

    auto ip_pd = dnnl::inner_product_forward::primitive_desc(
            ip_desc, ip_attr, engine_);

    // Set input-end memories
    auto ip_src_memory = dnnl::memory(ip_pd.src_desc(), engine_);
    dnnl::primitive_attr src_attr;
    src_attr.set_output_scales(src_mask, data_scales);
    auto src_reorder_pd
            = dnnl::reorder::primitive_desc(engine_, user_src_memory.get_desc(), engine_,
                    ip_src_memory.get_desc(), src_attr);
    auto src_reorder = dnnl::reorder(src_reorder_pd);
    net_.push_back(src_reorder);
    net_args_.push_back({{DNNL_ARG_SRC, user_src_memory},
                {DNNL_ARG_DST, ip_src_memory}});

    auto ip_weights_memory = dnnl::memory(ip_pd.weights_desc(), engine_);
    dnnl::primitive_attr weight_attr;
    weight_attr.set_output_scales(weight_mask, weight_scales);
    auto weight_reorder_pd
            = dnnl::reorder::primitive_desc(engine_, user_wei_memory.get_desc(), engine_,
                    ip_weights_memory.get_desc(), weight_attr);
    auto weight_reorder = dnnl::reorder(weight_reorder_pd);
    net_.push_back(weight_reorder);
    net_args_.push_back({{DNNL_ARG_SRC, user_wei_memory},
                {DNNL_ARG_DST, ip_weights_memory}});

    auto ip_bias_memory = dnnl::memory(ip_pd.bias_desc(), engine_);
    dnnl::primitive_attr bias_attr;
    bias_attr.set_output_scales(bias_mask, bias_scales);
    auto bias_reorder_pd
            = dnnl::reorder::primitive_desc(engine_, user_bia_memory.get_desc(), engine_,
                    ip_bias_memory.get_desc(), bias_attr);
    auto bias_reorder = dnnl::reorder(bias_reorder_pd);
    net_.push_back(bias_reorder);
    net_args_.push_back({{DNNL_ARG_SRC, user_bia_memory},
                {DNNL_ARG_DST, ip_bias_memory}});

    auto ip_dst_memory = dnnl::memory(ip_pd.dst_desc(), engine_);
    // end

    auto ip_prim = dnnl::inner_product_forward(ip_pd);
    net_.push_back(ip_prim);
    net_args_.push_back({{DNNL_ARG_SRC, ip_src_memory},
                {DNNL_ARG_WEIGHTS, ip_weights_memory},
                {DNNL_ARG_BIAS, ip_bias_memory},
                {DNNL_ARG_DST, ip_dst_memory}});

    auto user_dst_md = dnnl::memory::desc({ip_dst_tz, dt::f32, tag::ab});
    auto user_dst_memory = BindDNNLMemory(out_entry, user_dst_md);
    dnnl::primitive_attr dst_attr;
    dst_attr.set_output_scales(dst_mask, dst_scales);
    auto dst_reorder_pd
            = dnnl::reorder::primitive_desc(engine_, ip_dst_memory.get_desc(), engine_,
                    user_dst_memory.get_desc(), dst_attr);
    auto dst_reorder = dnnl::reorder(dst_reorder_pd);

    net_.push_back(dst_reorder);
    net_args_.push_back({{DNNL_ARG_SRC, ip_dst_memory},
                {DNNL_ARG_DST, user_dst_memory}});

    std::cout << "primitives created successfully" << std::endl;
  }

  void Qint8_Conv2d(const size_t& nid) {
    std::cout << "Qint8_Conv2d" << std::endl;
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();

    std::cout << "Node number of inputs: " << node.GetInputs().size() << std::endl;
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];

    JSONGraphNodeEntry out_entry(nid, 0);

#define GET_NODE_INFO(n) \
  int const##n##_json_node_entry_id = EntryID(node.GetInputs()[n].id_, 0); \
  std::cout << node.GetInputs()[n].id_ << std::endl;\
  std::cout << node.GetInputs()[n].index_ << std::endl;\
  std::cout << "const node json node entry id: " << const##n##_json_node_entry_id << std::endl; \
  auto const##n##_tensor_ptr = data_entry_[const##n##_json_node_entry_id]; \
  auto const##n##_ndim = const##n##_tensor_ptr->ndim; \
  std::cout << "data count: " << this->GetDataCount(*const##n##_tensor_ptr) << std::endl; \
  std::cout << "First elemt of const node: " << *((float*)const##n##_tensor_ptr->data) << '\n' << std::endl;

    // GET_NODE_INFO(0) // not const cannot get
    GET_NODE_INFO(1)
    GET_NODE_INFO(2)
    GET_NODE_INFO(3)
    GET_NODE_INFO(4)
    GET_NODE_INFO(5)
    GET_NODE_INFO(6)
    GET_NODE_INFO(7)
    GET_NODE_INFO(8)
    GET_NODE_INFO(9)
    // GET_NODE_INFO(10)

    std::cout << "nodes_ size: " << nodes_.size() << std::endl;
    std::cout << "node input size: " << node.GetInputs().size() << std::endl;
    // nodes_ is global, node is local
    dnnl::memory::dims conv_src_tz = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims conv_weights_tz = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims conv_dst_tz = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    dnnl::memory::dim out_channel = conv_weights_tz[0];
    dnnl::memory::dims conv_bias_tz = {out_channel};

    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");

    dnnl::memory::dims conv_strides = TransformStr2Dims(str_strides);
    dnnl::memory::dims conv_dilates = TransformStr2Dims(str_dilates, true);
    dnnl::memory::dims conv_padding = TransformStr2Dims(str_padding);

    // debug
    std::vector<float> data_scales;
    std::vector<float> weight_scales;
    std::vector<float> bias_scales = {1.0f};
    std::vector<float> dst_scales; // What is this?
    std::vector<float> conv_scales = {1.0f}; // what is this?

    for (uint32_t i = 0; i < this->GetDataCount(*const1_tensor_ptr); i++) {
      float data_scale = *((float*)const1_tensor_ptr->data + i);
      float weight_scale = *((float*)const4_tensor_ptr->data + i);
      // float conv_scale = *((float*)const7_tensor_ptr->data + i);
      data_scales.push_back(1.0f / data_scale);
      weight_scales.push_back(1.0f / weight_scale);
      dst_scales.push_back(data_scale * weight_scale);
    }

    std::cout << "data_scales: " << std::endl;
    for (auto e : data_scales) {
      std::cout << e << " ";
    }
    std::cout << std::endl;

    std::cout << "weight_scales: " << std::endl;
    for (auto e : weight_scales) {
      std::cout << e << " ";
    }
    std::cout << std::endl;

    std::cout << "dst_scales: " << std::endl;
    for (auto e : dst_scales) {
      std::cout << e << " ";
    }
    std::cout << std::endl;

    // for (uint32_t i = 0; i < this->GetDataCount(*const1_tensor_ptr); i++) {
    //   data_scales.push_back(*((float*)const1_tensor_ptr->data + i));
    // }

    // for (uint32_t i = 0; i < this->GetDataCount(*const4_tensor_ptr); i++) {
    //   weight_scales.push_back(*((float*)const4_tensor_ptr->data + i));
    // }

    // for (uint32_t i = 0; i < this->GetDataCount(*const1_tensor_ptr); i++) {
    //   dst_scales.push_back(1 / (data_scales[i] * weight_scales[i]));
    // }

    // for (uint32_t i = 0; i < this->GetDataCount(*const7_tensor_ptr); i++) {
    //   conv_scales.push_back(*((float*)const7_tensor_ptr->data + i));
    // }

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;
    const int dst_mask = 0;
    const int conv_mask = 0;

    auto user_src_md = dnnl::memory::desc({conv_src_tz, dt::f32, tag::nchw});
    auto user_wei_md = dnnl::memory::desc({conv_weights_tz, dt::f32, tag::iohw});
    auto user_bia_md = dnnl::memory::desc({conv_bias_tz, dt::f32, tag::x});

    auto user_src_memory = BindDNNLMemory(node.GetInputs()[0], user_src_md);
    auto user_wei_memory = BindDNNLMemory(node.GetInputs()[10], user_wei_md);
    auto user_bia_memory = dnnl::memory(user_bia_md, engine_);
    float bias[out_channel] = {0};
    write_to_dnnl_memory(bias, user_bia_memory, out_channel * sizeof(float));

    // debug
    dnnl::memory::dims conv_const_tz = {1};
    auto user_const_md = dnnl::memory::desc({conv_const_tz, dt::f32, tag::x});
    auto user_data_scale_memory = BindDNNLMemory(node.GetInputs()[1], user_const_md);
    auto user_data_min_memory = BindDNNLMemory(node.GetInputs()[2], user_const_md);
    auto user_data_max_memory = BindDNNLMemory(node.GetInputs()[3], user_const_md);
    auto user_weight_scale_memory = BindDNNLMemory(node.GetInputs()[4], user_const_md);
    auto user_weight_min_memory = BindDNNLMemory(node.GetInputs()[5], user_const_md);
    auto user_weight_max_memory = BindDNNLMemory(node.GetInputs()[6], user_const_md);
    auto user_conv_scale_memory = BindDNNLMemory(node.GetInputs()[7], user_const_md);
    auto user_conv_min_memory = BindDNNLMemory(node.GetInputs()[8], user_const_md);
    auto user_conv_max_memory = BindDNNLMemory(node.GetInputs()[9], user_const_md);
    // debug end

    auto conv_src_md = dnnl::memory::desc({conv_src_tz}, dt::u8, tag::any);
    auto conv_bias_md = dnnl::memory::desc({conv_bias_tz}, dt::s8, tag::any);
    auto conv_weights_md = dnnl::memory::desc({conv_weights_tz}, dt::s8, tag::any);
    auto conv_dst_md = dnnl::memory::desc({conv_dst_tz}, dt::s32, tag::any);

    auto conv_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward,
        dnnl::algorithm::convolution_direct, conv_src_md, conv_weights_md,
        conv_bias_md, conv_dst_md, conv_strides, conv_padding,
        conv_padding);

    dnnl::primitive_attr conv_attr;
    conv_attr.set_output_scales(conv_mask, conv_scales);
   
    // const float ops_scale = 1.f;
    // const float ops_alpha = 0.f; // relu negative slope
    // const float ops_beta = 0.f;
    // dnnl::post_ops ops;
    // ops.append_eltwise(ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
    // conv_attr.set_post_ops(ops);
    auto conv_prim_desc
            = dnnl::convolution_forward::primitive_desc(conv_desc, conv_attr, engine_);

    auto conv_src_memory = dnnl::memory(conv_prim_desc.src_desc(), engine_);
    dnnl::primitive_attr src_attr;
    src_attr.set_output_scales(src_mask, data_scales);
    auto src_reorder_pd
            = dnnl::reorder::primitive_desc(engine_, user_src_memory.get_desc(), engine_,
                    conv_src_memory.get_desc(), src_attr);
    auto src_reorder = dnnl::reorder(src_reorder_pd);
    net_.push_back(src_reorder);
    net_args_.push_back({{DNNL_ARG_SRC, user_src_memory},
                {DNNL_ARG_DST, conv_src_memory}});

    auto conv_wei_memory = dnnl::memory(conv_prim_desc.weights_desc(), engine_);
    dnnl::primitive_attr wei_attr;
    wei_attr.set_output_scales(weight_mask, weight_scales);
    auto wei_reorder_pd
            = dnnl::reorder::primitive_desc(engine_, user_wei_memory.get_desc(), engine_,
                    conv_wei_memory.get_desc(), wei_attr);
    auto wei_reorder = dnnl::reorder(wei_reorder_pd);
    net_.push_back(wei_reorder);
    net_args_.push_back({{DNNL_ARG_SRC, user_wei_memory},
                {DNNL_ARG_DST, conv_wei_memory}});

    auto conv_bia_memory = dnnl::memory(conv_prim_desc.bias_desc(), engine_);
    dnnl::primitive_attr bia_attr;
    bia_attr.set_output_scales(bias_mask, bias_scales);
    auto bia_reorder_pd
            = dnnl::reorder::primitive_desc(engine_, user_bia_memory.get_desc(), engine_,
                    conv_bia_memory.get_desc(), bia_attr);
    auto bia_reorder = dnnl::reorder(bia_reorder_pd);
    net_.push_back(bia_reorder);
    net_args_.push_back({{DNNL_ARG_SRC, user_bia_memory},
                {DNNL_ARG_DST, conv_bia_memory}});

    auto conv_dst_memory = dnnl::memory(conv_prim_desc.dst_desc(), engine_);

    // here
    auto conv_prim = dnnl::convolution_forward(conv_prim_desc);
    net_.push_back(conv_prim);
    net_args_.push_back({{DNNL_ARG_SRC, conv_src_memory},
                {DNNL_ARG_WEIGHTS, conv_wei_memory},
                {DNNL_ARG_BIAS, conv_bia_memory},
                {DNNL_ARG_DST, conv_dst_memory}});

    auto user_dst_md = dnnl::memory::desc({conv_dst_tz, dt::f32, tag::nchw});
    auto user_dst_memory = BindDNNLMemory(out_entry, user_dst_md);
    dnnl::primitive_attr dst_attr;
    dst_attr.set_output_scales(dst_mask, dst_scales);
    auto dst_reorder_pd
            = dnnl::reorder::primitive_desc(engine_, conv_dst_memory.get_desc(), engine_,
                    user_dst_memory.get_desc(), dst_attr);
    auto dst_reorder = dnnl::reorder(dst_reorder_pd);

    net_.push_back(dst_reorder);
    net_args_.push_back({{DNNL_ARG_SRC, conv_dst_memory},
                {DNNL_ARG_DST, user_dst_memory}});
  }

  void Convolution(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    dnnl::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    dnnl::memory::dim channels =
        node.GetAttr<std::vector<std::string>>("channels")[0] != ""
            ? std::stoi(node.GetAttr<std::vector<std::string>>("channels")[0])
            : out_shape[1];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    std::string data_layout = node.GetAttr<std::vector<std::string>>("data_layout")[0];
    std::string kernel_layout = node.GetAttr<std::vector<std::string>>("kernel_layout")[0];

    // Check layout.
    if (layout_dict.find(data_layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported data layout for conv: " << data_layout;
    }

    if (layout_dict.find(kernel_layout) == layout_dict.end()) {
      layout_dict.insert({kernel_layout, tag::any});
      LOG(WARNING) << "Unregistered kernel layout for conv: " << kernel_layout
                   << ", transfer to tag::any";
    }

    // Memory shapes.
    dnnl::memory::dims src_dims = TransDims2Plain(input_shape, data_layout);
    dnnl::memory::dims weights_dims_ = TransDims2Plain(weight_shape, kernel_layout);
    dnnl::memory::dims bias_dims = {channels};
    dnnl::memory::dims strides_dims = TransformStr2Dims(str_strides);
    dnnl::memory::dims dilates_dims = TransformStr2Dims(str_dilates, true);
    dnnl::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l);
    dnnl::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r);
    dnnl::memory::dims dst_dims = src_dims;
    dst_dims[1] = channels;
    weights_dims_[0] = channels;
    for (size_t i = 2; i < src_dims.size(); i++) {
      dnnl::memory::dim K = weights_dims_[i];
      dnnl::memory::dim S = strides_dims[i - 2];
      dnnl::memory::dim D = dilates_dims[i - 2];
      dnnl::memory::dim PL = padding_dims_l[i - 2];
      dnnl::memory::dim PR = padding_dims_r[i - 2];
      dnnl::memory::dim DK = 1 + (K - 1) * (D + 1);
      dst_dims[i] = (src_dims[i] - DK + PL + PR) / S + 1;
    }

    dnnl::memory::dims weights_dims = weights_dims_;
    if (groups > 1) {
      weights_dims = {groups, channels / groups, src_dims[1] / groups};
      weights_dims.insert(weights_dims.end(), weights_dims_.begin() + 2, weights_dims_.end());
      if (kernel_layout == "OIHW") {
        kernel_layout.insert(0, "G");
      }
    }

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::f32, layout_dict[data_layout]);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, layout_dict[kernel_layout]);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::any);

    // Conv description.
    auto conv_desc =
        has_bias ? dnnl::convolution_forward::desc(
                       dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                       conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, strides_dims,
                       dilates_dims, padding_dims_l, padding_dims_r)
                 : dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                                   dnnl::algorithm::convolution_direct, conv_src_md,
                                                   conv_weights_md, conv_dst_md, strides_dims,
                                                   dilates_dims, padding_dims_l, padding_dims_r);

    // Enable elementwise post-ops.
    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv_prim_desc);
    net_.push_back(conv);

    // Data memory.
    auto conv_src_memory = BindDNNLMemory(data_entry, conv_src_md);

    // Weight memory.
    auto conv_weights_memory = BindDNNLMemory(weight_entry, conv_prim_desc.weights_desc());

    // Output memory.
    auto conv_dst_memory = BindDNNLMemory(out_entry, conv_prim_desc.dst_desc());

    // Bias memory.
    auto conv_bias_memory = dnnl::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, conv_bias_memory);

      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, conv_src_memory},
                           {DNNL_ARG_WEIGHTS, conv_weights_memory},
                           {DNNL_ARG_BIAS, conv_bias_memory},
                           {DNNL_ARG_DST, conv_dst_memory}});
    } else {
      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, conv_src_memory},
                           {DNNL_ARG_WEIGHTS, conv_weights_memory},
                           {DNNL_ARG_DST, conv_dst_memory}});
    }
  }

  void Deconvolution(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    dnnl::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    dnnl::memory::dim channels =
        node.GetAttr<std::vector<std::string>>("channels")[0] != ""
            ? std::stoi(node.GetAttr<std::vector<std::string>>("channels")[0])
            : out_shape[1];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    std::vector<std::string> str_out_padding =
        node.GetAttr<std::vector<std::string>>("output_padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    std::string data_layout = node.GetAttr<std::vector<std::string>>("data_layout")[0];
    std::string kernel_layout = node.GetAttr<std::vector<std::string>>("kernel_layout")[0];

    // Check layout.
    if (layout_dict.find(data_layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported data layout for deconv: " << data_layout;
    }

    if (layout_dict.find(kernel_layout) == layout_dict.end()) {
      layout_dict.insert({kernel_layout, tag::any});
      LOG(WARNING) << "Unregistered kernel layout for deconv: " << data_layout
                   << ", transfer to tag::any";
    }

    // Memory shapes.
    dnnl::memory::dims src_dims = TransDims2Plain(input_shape, data_layout);
    dnnl::memory::dims weights_dims_ = TransDims2Plain(weight_shape, kernel_layout);
    // legalize shape IOHW with layout OIHW
    if (weights_dims_[0] == src_dims[1] && weights_dims_[1] == channels) {
      std::swap(weights_dims_[0], weights_dims_[1]);
      if (kernel_layout.find("OI") == 0) {
        kernel_layout.replace(kernel_layout.find("OI"), 2, "IO");
      }
    }
    dnnl::memory::dims bias_dims = {channels};
    dnnl::memory::dims strides_dims = TransformStr2Dims(str_strides);
    dnnl::memory::dims dilates_dims = TransformStr2Dims(str_dilates, true);
    dnnl::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l);
    dnnl::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r);
    dnnl::memory::dims out_padding = TransformStr2Dims(str_out_padding);
    dnnl::memory::dims dst_dims = src_dims;
    dst_dims[1] = channels;
    for (size_t i = 2; i < src_dims.size(); i++) {
      dnnl::memory::dim K = weights_dims_[i];
      dnnl::memory::dim S = strides_dims[i - 2];
      dnnl::memory::dim D = dilates_dims[i - 2];
      dnnl::memory::dim PL = padding_dims_l[i - 2];
      dnnl::memory::dim PR = padding_dims_r[i - 2];
      dnnl::memory::dim OP = out_padding[i - 2];
      dnnl::memory::dim DK = 1 + (K - 1) * (D + 1);
      dst_dims[i] = S * (src_dims[i] - 1) + DK - PL - PR + OP;
    }

    dnnl::memory::dims weights_dims = weights_dims_;
    if (groups > 1) {
      weights_dims = {groups, channels / groups, src_dims[1] / groups};
      weights_dims.insert(weights_dims.end(), weights_dims_.begin() + 2, weights_dims_.end());
    }

    // Memory descriptions.
    auto deconv_src_md = dnnl::memory::desc(src_dims, dt::f32, layout_dict[data_layout]);
    auto deconv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, layout_dict[kernel_layout]);
    auto deconv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::any);
    auto deconv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::any);

    // Transposed covn2d description.
    auto deconv_desc =
        has_bias ? dnnl::deconvolution_forward::desc(
                       dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                       deconv_src_md, deconv_weights_md, deconv_bias_md, deconv_dst_md,
                       strides_dims, dilates_dims, padding_dims_l, padding_dims_r)
                 : dnnl::deconvolution_forward::desc(
                       dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                       deconv_src_md, deconv_weights_md, deconv_dst_md, strides_dims, dilates_dims,
                       padding_dims_l, padding_dims_r);

    // Enable elementwise post-ops.
    auto deconv_prim_desc = dnnl::deconvolution_forward::primitive_desc(deconv_desc, attr, engine_);

    // Push to the network.
    auto deconv = dnnl::deconvolution_forward(deconv_prim_desc);
    net_.push_back(deconv);

    // Data memory.
    auto deconv_src_memory = BindDNNLMemory(data_entry, deconv_src_md);

    // Weight memory.
    auto deconv_weights_memory = BindDNNLMemory(weight_entry, deconv_prim_desc.weights_desc());

    // Output memory.
    auto deconv_dst_memory = BindDNNLMemory(out_entry, deconv_prim_desc.dst_desc());

    // Bias memory.
    auto deconv_bias_memory = dnnl::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, deconv_bias_memory);

      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, deconv_src_memory},
                           {DNNL_ARG_WEIGHTS, deconv_weights_memory},
                           {DNNL_ARG_BIAS, deconv_bias_memory},
                           {DNNL_ARG_DST, deconv_dst_memory}});
    } else {
      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, deconv_src_memory},
                           {DNNL_ARG_WEIGHTS, deconv_weights_memory},
                           {DNNL_ARG_DST, deconv_dst_memory}});
    }
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    dnnl::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    dnnl::memory::dim OC = out_shape[1];

    // Memory shapes.
    dnnl::memory::dims data_dims = input_shape;
    dnnl::memory::dims weight_dims = weight_shape;
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = out_shape;

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);

    // Enable elementwise post-ops.
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, attr, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);

    // Bias memory.
    auto bias_memory = dnnl::memory(bias_md, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, bias_memory);
    } else {
      float bias[OC] = {0};
      write_to_dnnl_memory(bias, bias_memory, OC * sizeof(float));
    }

    // Output memory.
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

  void Pooling(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    std::vector<std::string> str_kernel = node.GetAttr<std::vector<std::string>>("pool_size");
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::string layout = node.GetAttr<std::vector<std::string>>("layout")[0];

    // Check layout.
    if (layout_dict.find(layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported layout for pooling: " << layout;
    }

    // Attributes related to AvgPool
    if (algo == dnnl::algorithm::pooling_avg) {
      int int_countpad = std::stoi(node.GetAttr<std::vector<std::string>>("count_include_pad")[0]);
      bool count_include_pad = int_countpad != 0 ? true : false;
      algo = count_include_pad ? dnnl::algorithm::pooling_avg_include_padding
                               : dnnl::algorithm::pooling_avg_exclude_padding;
    }

    dnnl::memory::dims src_dims = TransDims2Plain(input_shape, layout);
    dnnl::memory::dims dst_dims = TransDims2Plain(out_shape, layout);
    dnnl::memory::dims kernel_dims = TransformStr2Dims(str_kernel);
    dnnl::memory::dims strides_dims = TransformStr2Dims(str_strides);
    dnnl::memory::dims dilates_dims = TransformStr2Dims(str_dilates, true);
    dnnl::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l);
    dnnl::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r);

    // Memory descriptions.
    auto pool_src_md = dnnl::memory::desc(src_dims, dt::f32, layout_dict[layout]);
    auto pool_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::any);

    // Pooling description.
    auto pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference, algo,
                                                 pool_src_md, pool_dst_md, strides_dims,
                                                 kernel_dims, padding_dims_l, padding_dims_r);

    auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(pool_desc, engine_, true);
    auto pool = dnnl::pooling_forward(pool_prim_desc);
    net_.push_back(pool);

    // Memories.
    auto pool2d_src_memory = BindDNNLMemory(data_entry, pool_src_md);

    auto pool2d_dst_memory = BindDNNLMemory(out_entry, pool_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({{DNNL_ARG_SRC, pool2d_src_memory}, {DNNL_ARG_DST, pool2d_dst_memory}});
  }

  void Eltwise(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    auto algo = elt_name2algo[op_name];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(shape, dt::f32);
    float alpha = 0., beta = 0.;
    if (op_name == "clip") {
      alpha = std::stof(node.GetAttr<std::vector<std::string>>("a_min")[0]);
      beta = std::stof(node.GetAttr<std::vector<std::string>>("a_max")[0]);
    } else if (op_name == "nn.leaky_relu") {
      alpha = std::stof(node.GetAttr<std::vector<std::string>>("alpha")[0]);
    }

    auto elt_desc =
        dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algo, data_md, alpha, beta);
    auto elt_prim_desc = dnnl::eltwise_forward::primitive_desc(elt_desc, engine_);
    ICHECK(data_md == elt_prim_desc.dst_desc());

    auto elt = dnnl::eltwise_forward(elt_prim_desc);
    net_.push_back(elt);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    if (axis < 0) {
      axis = shape.size() + axis;
    }
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(shape, dt::f32);

    auto softmax_desc =
        dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference, data_md, axis);
    auto softmax_prim_desc = dnnl::softmax_forward::primitive_desc(softmax_desc, engine_);
    ICHECK(data_md == softmax_prim_desc.dst_desc());

    auto softmax = dnnl::softmax_forward(softmax_prim_desc);
    net_.push_back(softmax);

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
