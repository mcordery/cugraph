/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace raft::neighbors::detail::faiss_select {

template <typename T>
struct Comparator {
  __device__ static inline bool lt(T a, T b) { return a < b; }

  __device__ static inline bool gt(T a, T b) { return a > b; }
};

template <>
struct Comparator<half> {
  __device__ static inline bool lt(half a, half b) { return __hlt(a, b); }

  __device__ static inline bool gt(half a, half b) { return __hgt(a, b); }
};

}  // namespace raft::neighbors::detail::faiss_select
