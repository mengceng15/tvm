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
 * \file src/relay/backend/contrib/dnnl/query_layout.cc
 * \brief layout auto-query func.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>
#include "dnnl.hpp"

#include "../../utils.h"

using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;

namespace tvm {
namespace relay {
namespace contrib {

template <typename T, typename U>
inline void array_set(T *arr, const U &val, size_t size) {
    for (size_t i = 0; i < size; ++i)
        arr[i] = static_cast<T>(val);
}

template <typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

void compute_blocks(dims_t &blocks, const dnnl::memory::desc *md) {
        using format_kind_t = dnnl_format_kind_t;
        const format_kind_t blocked = dnnl_blocked;
        if (!(md->data.format_kind==blocked)) {
            array_set(blocks, 0, md->data.ndims);
            return;
        }

        array_set(blocks, 1, md->data.ndims);

        const auto &bd = md->data.format_desc.blocking;
        for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
            blocks[bd.inner_idxs[iblk]] *= bd.inner_blks[iblk];
    }

inline bool has_runtime_strides(const dnnl::memory::desc *md) {
        using format_kind_t = dnnl_format_kind_t;
        const format_kind_t blocked = dnnl_blocked;
        if (!(md->data.format_kind==blocked)) return false;
        for (int d = 0; d < md->data.ndims; ++d)
            if (md->data.format_desc.blocking.strides[d] == DNNL_RUNTIME_DIM_VAL) return true;
        return false;
    }

template <typename T>
inline void swap(T &t1, T &t2) {
    T tmp(t1);
    t1 = t2;
    t2 = tmp;
}

template <typename T, typename U, typename F>
inline void simultaneous_sort(
        T *vals, T *vals_2nd_level, U *keys, size_t size, F comparator) {
    if (size == 0) return;

    for (size_t i = 0; i < size - 1; ++i) {
        bool swapped = false;

        for (size_t j = 0; j < size - i - 1; j++) {
            auto res = comparator(vals[j], vals[j + 1]);
            if (res == 0)
                res = comparator(vals_2nd_level[j], vals_2nd_level[j + 1]);

            if (res > 0) {
                swap(vals[j], vals[j + 1]);
                swap(vals_2nd_level[j], vals_2nd_level[j + 1]);
                swap(keys[j], keys[j + 1]);
                swapped = true;
            }
        }

        if (swapped == false) break;
    }
}

std::string md2fmt_tag_str(const dnnl::memory::desc *md) {
    // memory_desc_wrapper mdw(md);

    const auto &blk = md->data.format_desc.blocking;//mdw.blocking_desc();

    dims_t blocks = {0};
    compute_blocks(blocks, md);

    char dim_chars[DNNL_MAX_NDIMS + 1];

    dims_t ou_blocks = {0};
    array_copy(ou_blocks, md->data.padded_dims, md->data.ndims);

    bool plain = true;
    for (int d = 0; d < md->data.ndims; ++d) {
        dim_chars[d] = (blocks[d] == 1 ? 'a' : 'A') + (char)d;
        if (blocks[d] != 1) plain = false;
        ou_blocks[d] /= blocks[d];
    }

    // Can't report meaningful tag for runtime dimensions.
    if (has_runtime_strides(md)) return "*";

    dims_t strides;
    array_copy(strides, blk.strides, md->data.ndims);

    simultaneous_sort(strides, ou_blocks, dim_chars, md->data.ndims,
            [](dim_t a, dim_t b) { return b - a; });

    dim_chars[md->data.ndims] = '\0';

    std::string s(dim_chars);

    if (!plain) {
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            char c = ('a' + (char)blk.inner_idxs[iblk]);
            s += (std::to_string(blk.inner_blks[iblk]) + c);
        }
    }
    return s;
}

std::string AutoQuery(int N,int IC,int KH,int KW,int OC,int SH,int SW,int PH_L,int PH_R,int PW_L,int PW_R,int OH,int OW) {//int *shapes, struct StructFormat* res
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream s(eng);
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    
    auto IH = OH * SH - PH_L -PH_R + KH - 1;
    auto IW = OW * SW - PW_L -PW_R + KW - 1;

    const dnnl::memory::dim batch = N;

    dnnl::memory::dims conv1_src_tz = {batch, IC, IH, IW};//{batch, 3, 227, 227};
    dnnl::memory::dims conv1_weights_tz = {OC, IC, KH, KW};//{96, 3, 11, 11};
    dnnl::memory::dims conv1_bias_tz = {OC};//{96};
    dnnl::memory::dims conv1_dst_tz = {batch, OC, OH, OW};//{batch, 96, 55, 55};
    dnnl::memory::dims conv1_strides = {SH, SW};
    dnnl::memory::dims conv1_padding_l = {PH_L, PW_L};
    dnnl::memory::dims conv1_padding_r = {PH_R, PW_R};

    auto conv1_src_md = dnnl::memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_bias_md = dnnl::memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = dnnl::memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_dst_md = dnnl::memory::desc({conv1_dst_tz}, dt::f32, tag::any);
    //[Create convolution memory descriptors]

    auto conv1_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
            conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding_l,
            conv1_padding_r);

    auto conv1_prim_desc = dnnl::convolution_forward::primitive_desc(conv1_desc, eng);

    auto src_format = conv1_prim_desc.src_desc();//.data;
    auto weights_format = conv1_prim_desc.weights_desc();//.data;
    // auto bias_format = conv1_prim_desc.bias_desc();
    auto dst_format = conv1_prim_desc.dst_desc();//.data;
    std::string src_df, weight_df, dst_df;

    src_df = md2fmt_tag_str(&src_format);
    weight_df = md2fmt_tag_str(&weights_format);
    dst_df = md2fmt_tag_str(&dst_format);
    std::string res = src_df + "," + weight_df + "," + dst_df;
    return res;
}

std::string AutoQuery_matmul(int M, int K, int N) {//int *shapes, struct StructFormat* res
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream s(eng);
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    // const dnnl::memory::dim batch = N;

    dnnl::memory::dims matmul1_src_tz = {M, K};
    dnnl::memory::dims matmul1_weights_tz = {K, N};
    dnnl::memory::dims matmul1_bias_tz = {M, N};
    dnnl::memory::dims matmul1_dst_tz = {M, N};

    auto matmul1_src_md = dnnl::memory::desc({matmul1_src_tz}, dt::f32, tag::any);
    auto matmul1_weights_md = dnnl::memory::desc({matmul1_weights_tz}, dt::f32, tag::any);
    auto matmul1_bias_md = dnnl::memory::desc({matmul1_bias_tz}, dt::f32, tag::any);
    auto matmul1_dst_md = dnnl::memory::desc({matmul1_dst_tz}, dt::f32, tag::any);

    auto matmul1_d = dnnl::matmul::desc(matmul1_src_md, matmul1_weights_md,
            matmul1_bias_md, matmul1_dst_md);
    auto matmul1_pd = dnnl::matmul::primitive_desc(matmul1_d, eng);

    auto src_format = matmul1_pd.src_desc();//.data;
    auto weights_format = matmul1_pd.weights_desc();//.data;
    auto bias_format = matmul1_pd.bias_desc();
    auto dst_format = matmul1_pd.dst_desc();//.data;
    std::string src_df, weight_df, bias_df, dst_df;

    src_df = md2fmt_tag_str(&src_format);
    weight_df = md2fmt_tag_str(&weights_format);
    bias_df = md2fmt_tag_str(&bias_format);
    dst_df = md2fmt_tag_str(&dst_format);
    std::string res = src_df + "," + weight_df + "," + bias_df + "," + dst_df;
    return res;
}

std::string AutoQuery_innerproduct(int B, int IC, int OC) {//int *shapes, struct StructFormat* res
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream s(eng);
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    // const dnnl::memory::dim batch = N;

    dnnl::memory::dims dense1_src_tz = {B, IC};
    dnnl::memory::dims dense1_weights_tz = {OC, IC};
    dnnl::memory::dims dense1_bias_tz = {OC};
    dnnl::memory::dims dense1_dst_tz = {B, OC};

    auto dense1_src_md = dnnl::memory::desc({dense1_src_tz}, dt::f32, tag::any);
    auto dense1_weights_md = dnnl::memory::desc({dense1_weights_tz}, dt::f32, tag::any);
    auto dense1_bias_md = dnnl::memory::desc({dense1_bias_tz}, dt::f32, tag::any);
    auto dense1_dst_md = dnnl::memory::desc({dense1_dst_tz}, dt::f32, tag::any);

    auto dense1_d = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference,
            dense1_src_md, dense1_weights_md, dense1_bias_md, dense1_dst_md);
    auto dense1_pd = dnnl::inner_product_forward::primitive_desc(dense1_d, eng);

    auto src_format = dense1_pd.src_desc();//.data;
    auto weights_format = dense1_pd.weights_desc();//.data;
    auto bias_format = dense1_pd.bias_desc();
    auto dst_format = dense1_pd.dst_desc();//.data;
    std::string src_df, weight_df, bias_df, dst_df;

    src_df = md2fmt_tag_str(&src_format);
    weight_df = md2fmt_tag_str(&weights_format);
    bias_df = md2fmt_tag_str(&bias_format);
    dst_df = md2fmt_tag_str(&dst_format);
    std::string res = src_df + "," + weight_df + "," + bias_df + "," + dst_df;
    return res;
}

TVM_REGISTER_GLOBAL("relay.ir.AutoQuery").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = AutoQuery(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]);
});

TVM_REGISTER_GLOBAL("relay.ir.AutoQuery_matmul").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = AutoQuery_matmul(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("relay.ir.AutoQuery_innerproduct").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = AutoQuery_innerproduct(args[0], args[1], args[2]);
});
}  // namespace contrib
}  // namespace relay
}  // namespace tvm