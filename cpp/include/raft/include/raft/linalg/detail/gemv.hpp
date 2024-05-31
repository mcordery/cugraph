/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cublas_wrappers.hpp"

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resources.hpp>

#include <hipblas.h>

namespace raft {
namespace linalg {
namespace detail {

template <typename math_t, bool DevicePointerMode = false>
void gemv(raft::resources const& handle,
          const bool trans_a,
          const int m,
          const int n,
          const math_t* alpha,
          const math_t* A,
          const int lda,
          const math_t* x,
          const int incx,
          const math_t* beta,
          math_t* y,
          const int incy,
          hipStream_t stream)
{
  hipblasHandle_t hipblas.h = resource::get_cublas_handle(handle);
  detail::cublas_device_pointer_mode<DevicePointerMode> pmode(cublas_h);
  RAFT_CUBLAS_TRY(detail::cublasgemv(cublas_h,
                                     trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                     m,
                                     n,
                                     alpha,
                                     A,
                                     lda,
                                     x,
                                     incx,
                                     beta,
                                     y,
                                     incy,
                                     stream));
}

template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows,
          const int n_cols,
          const math_t* x,
          const int incx,
          math_t* y,
          const int incy,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          hipStream_t stream)
{
  gemv(handle, trans_a, n_rows, n_cols, &alpha, A, n_rows, x, incx, &beta, y, incy, stream);
}

template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          hipStream_t stream)
{
  gemv(handle, A, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, stream);
}

template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          hipStream_t stream)
{
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);

  gemv(handle, A, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, stream);
}

template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const int lda,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          hipStream_t stream)
{
  hipblasHandle_t hipblas.h = resource::get_cublas_handle(handle);
  hipblasOperation_t op_a  = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  RAFT_CUBLAS_TRY(
    cublasgemv(cublas_h, op_a, n_rows_a, n_cols_a, &alpha, A, lda, x, 1, &beta, y, 1, stream));
}

template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const int lda,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          hipStream_t stream)
{
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  gemv(handle, A, n_rows_a, n_cols_a, lda, x, y, trans_a, alpha, beta, stream);
}

};  // namespace detail
};  // namespace linalg
};  // namespace raft
