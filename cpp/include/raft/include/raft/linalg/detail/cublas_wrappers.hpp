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

#include <raft/core/cublas_macros.hpp>
#include <raft/core/error.hpp>

#include <hipblas.h>

#include <cstdint>

namespace raft {
namespace linalg {
namespace detail {

/**
 * Assuming the default HIPBLAS_POINTER_MODE_HOST, change it to host or device mode
 * temporary for the lifetime of this object.
 */
template <bool DevicePointerMode = false>
class cublas_device_pointer_mode {
 public:
  explicit cublas_device_pointer_mode(hipblasHandle_t handle) : handle_(handle)
  {
    if constexpr (DevicePointerMode) {
      RAFT_CUBLAS_TRY(hipblasSetPointerMode(handle_, HIPBLAS_POINTER_MODE_DEVICE));
    }
  }
  auto operator=(const cublas_device_pointer_mode&) -> cublas_device_pointer_mode& = delete;
  auto operator=(cublas_device_pointer_mode&&) -> cublas_device_pointer_mode&      = delete;
  static auto operator new(std::size_t) -> void*                                   = delete;
  static auto operator new[](std::size_t) -> void*                                 = delete;

  ~cublas_device_pointer_mode()
  {
    if constexpr (DevicePointerMode) {
      RAFT_CUBLAS_TRY_NO_THROW(hipblasSetPointerMode(handle_, HIPBLAS_POINTER_MODE_HOST));
    }
  }

 private:
  hipblasHandle_t handle_ = nullptr;
};

/**
 * @defgroup Axpy cublas ax+y operations
 * @{
 */
template <typename T>
hipblasStatus_t cublasaxpy(hipblasHandle_t handle,
                           int n,
                           const T* alpha,
                           const T* x,
                           int incx,
                           T* y,
                           int incy,
                           hipStream_t stream);

template <>
inline hipblasStatus_t cublasaxpy(hipblasHandle_t handle,
                                  int n,
                                  const float* alpha,
                                  const float* x,
                                  int incx,
                                  float* y,
                                  int incy,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
inline hipblasStatus_t cublasaxpy(hipblasHandle_t handle,
                                  int n,
                                  const double* alpha,
                                  const double* x,
                                  int incx,
                                  double* y,
                                  int incy,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}
/** @} */

/**
 * @defgroup cublas swap operations
 * @{
 */
template <typename T>
hipblasStatus_t cublasSwap(
  hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy, hipStream_t stream);

template <>
inline hipblasStatus_t cublasSwap(
  hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSswap(handle, n, x, incx, y, incy);
}

template <>
inline hipblasStatus_t cublasSwap(
  hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDswap(handle, n, x, incx, y, incy);
}

/** @} */

/**
 * @defgroup cublas copy operations
 * @{
 */
template <typename T>
hipblasStatus_t cublasCopy(
  hipblasHandle_t handle, int n, const T* x, int incx, T* y, int incy, hipStream_t stream);

template <>
inline hipblasStatus_t cublasCopy(
  hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy, hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasScopy(handle, n, x, incx, y, incy);
}
template <>
inline hipblasStatus_t cublasCopy(
  hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy, hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDcopy(handle, n, x, incx, y, incy);
}
/** @} */

/**
 * @defgroup gemv cublas gemv calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgemv(hipblasHandle_t handle,
                           hipblasOperation_t transA,
                           int m,
                           int n,
                           const T* alfa,
                           const T* A,
                           int lda,
                           const T* x,
                           int incx,
                           const T* beta,
                           T* y,
                           int incy,
                           hipStream_t stream);

template <>
inline hipblasStatus_t cublasgemv(hipblasHandle_t handle,
                                  hipblasOperation_t transA,
                                  int m,
                                  int n,
                                  const float* alfa,
                                  const float* A,
                                  int lda,
                                  const float* x,
                                  int incx,
                                  const float* beta,
                                  float* y,
                                  int incy,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSgemv(handle, transA, m, n, alfa, A, lda, x, incx, beta, y, incy);
}

template <>
inline hipblasStatus_t cublasgemv(hipblasHandle_t handle,
                                  hipblasOperation_t transA,
                                  int m,
                                  int n,
                                  const double* alfa,
                                  const double* A,
                                  int lda,
                                  const double* x,
                                  int incx,
                                  const double* beta,
                                  double* y,
                                  int incy,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDgemv(handle, transA, m, n, alfa, A, lda, x, incx, beta, y, incy);
}
/** @} */

/**
 * @defgroup ger cublas a(x*y.T) + A calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasger(hipblasHandle_t handle,
                          int m,
                          int n,
                          const T* alpha,
                          const T* x,
                          int incx,
                          const T* y,
                          int incy,
                          T* A,
                          int lda,
                          hipStream_t stream);
template <>
inline hipblasStatus_t cublasger(hipblasHandle_t handle,
                                 int m,
                                 int n,
                                 const float* alpha,
                                 const float* x,
                                 int incx,
                                 const float* y,
                                 int incy,
                                 float* A,
                                 int lda,
                                 hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline hipblasStatus_t cublasger(hipblasHandle_t handle,
                                 int m,
                                 int n,
                                 const double* alpha,
                                 const double* x,
                                 int incx,
                                 const double* y,
                                 int incy,
                                 double* A,
                                 int lda,
                                 hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
/** @} */

/**
 * @defgroup gemm cublas gemm calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgemm(hipblasHandle_t handle,
                           hipblasOperation_t transA,
                           hipblasOperation_t transB,
                           int m,
                           int n,
                           int k,
                           const T* alfa,
                           const T* A,
                           int lda,
                           const T* B,
                           int ldb,
                           const T* beta,
                           T* C,
                           int ldc,
                           hipStream_t stream);

template <>
inline hipblasStatus_t cublasgemm(hipblasHandle_t handle,
                                  hipblasOperation_t transA,
                                  hipblasOperation_t transB,
                                  int m,
                                  int n,
                                  int k,
                                  const float* alfa,
                                  const float* A,
                                  int lda,
                                  const float* B,
                                  int ldb,
                                  const float* beta,
                                  float* C,
                                  int ldc,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSgemm(handle, transA, transB, m, n, k, alfa, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline hipblasStatus_t cublasgemm(hipblasHandle_t handle,
                                  hipblasOperation_t transA,
                                  hipblasOperation_t transB,
                                  int m,
                                  int n,
                                  int k,
                                  const double* alfa,
                                  const double* A,
                                  int lda,
                                  const double* B,
                                  int ldb,
                                  const double* beta,
                                  double* C,
                                  int ldc,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDgemm(handle, transA, transB, m, n, k, alfa, A, lda, B, ldb, beta, C, ldc);
}
/** @} */

/**
 * @defgroup gemmbatched cublas gemmbatched calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgemmBatched(hipblasHandle_t handle,  // NOLINT
                                  hipblasOperation_t transa,
                                  hipblasOperation_t transb,
                                  int m,
                                  int n,
                                  int k,
                                  const T* alpha,
                                  const T* const Aarray[],  // NOLINT
                                  int lda,
                                  const T* const Barray[],  // NOLINT
                                  int ldb,
                                  const T* beta,
                                  T* Carray[],  // NOLINT
                                  int ldc,
                                  int batchCount,
                                  hipStream_t stream);

template <>
inline hipblasStatus_t cublasgemmBatched(  // NOLINT
  hipblasHandle_t handle,
  hipblasOperation_t transa,
  hipblasOperation_t transb,
  int m,
  int n,
  int k,
  const float* alpha,
  const float* const Aarray[],  // NOLINT
  int lda,
  const float* const Barray[],  // NOLINT
  int ldb,
  const float* beta,
  float* Carray[],  // NOLINT
  int ldc,
  int batchCount,
  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSgemmBatched(handle,
                             transa,
                             transb,
                             m,
                             n,
                             k,
                             alpha,
                             Aarray,
                             lda,
                             Barray,
                             ldb,
                             beta,
                             Carray,
                             ldc,
                             batchCount);
}

template <>
inline hipblasStatus_t cublasgemmBatched(  // NOLINT
  hipblasHandle_t handle,
  hipblasOperation_t transa,
  hipblasOperation_t transb,
  int m,
  int n,
  int k,
  const double* alpha,
  const double* const Aarray[],  // NOLINT
  int lda,
  const double* const Barray[],  // NOLINT
  int ldb,
  const double* beta,
  double* Carray[],  // NOLINT
  int ldc,
  int batchCount,
  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDgemmBatched(handle,
                             transa,
                             transb,
                             m,
                             n,
                             k,
                             alpha,
                             Aarray,
                             lda,
                             Barray,
                             ldb,
                             beta,
                             Carray,
                             ldc,
                             batchCount);
}
/** @} */

/**
 * @defgroup gemmbatched cublas gemmbatched calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgemmStridedBatched(  // NOLINT
  hipblasHandle_t handle,
  hipblasOperation_t transa,
  hipblasOperation_t transb,
  int m,
  int n,
  int k,
  const T* alpha,
  const T* const Aarray,
  int lda,
  int64_t strideA,
  const T* const Barray,
  int ldb,
  int64_t strideB,
  const T* beta,
  T* Carray,
  int ldc,
  int64_t strideC,
  int batchCount,
  hipStream_t stream);

template <>
inline hipblasStatus_t cublasgemmStridedBatched(  // NOLINT
  hipblasHandle_t handle,
  hipblasOperation_t transa,
  hipblasOperation_t transb,
  int m,
  int n,
  int k,
  const float* alpha,
  const float* const Aarray,
  int lda,
  int64_t strideA,
  const float* const Barray,
  int ldb,
  int64_t strideB,
  const float* beta,
  float* Carray,
  int ldc,
  int64_t strideC,
  int batchCount,
  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSgemmStridedBatched(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    Aarray,
                                    lda,
                                    strideA,
                                    Barray,
                                    ldb,
                                    strideB,
                                    beta,
                                    Carray,
                                    ldc,
                                    strideC,
                                    batchCount);
}

template <>
inline hipblasStatus_t cublasgemmStridedBatched(  // NOLINT
  hipblasHandle_t handle,
  hipblasOperation_t transa,
  hipblasOperation_t transb,
  int m,
  int n,
  int k,
  const double* alpha,
  const double* const Aarray,
  int lda,
  int64_t strideA,
  const double* const Barray,
  int ldb,
  int64_t strideB,
  const double* beta,
  double* Carray,
  int ldc,
  int64_t strideC,
  int batchCount,
  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDgemmStridedBatched(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    Aarray,
                                    lda,
                                    strideA,
                                    Barray,
                                    ldb,
                                    strideB,
                                    beta,
                                    Carray,
                                    ldc,
                                    strideC,
                                    batchCount);
}
/** @} */

/**
 * @defgroup solverbatched cublas getrf/gettribatched calls
 * @{
 */

template <typename T>
hipblasStatus_t cublasgetrfBatched(hipblasHandle_t handle,
                                   int n,         // NOLINT
                                   T* const A[],  // NOLINT
                                   int lda,
                                   int* P,
                                   int* info,
                                   int batchSize,
                                   hipStream_t stream);

template <>
inline hipblasStatus_t cublasgetrfBatched(hipblasHandle_t handle,  // NOLINT
                                          int n,
                                          float* const A[],  // NOLINT
                                          int lda,
                                          int* P,
                                          int* info,
                                          int batchSize,
                                          hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

template <>
inline hipblasStatus_t cublasgetrfBatched(hipblasHandle_t handle,  // NOLINT
                                          int n,
                                          double* const A[],  // NOLINT
                                          int lda,
                                          int* P,
                                          int* info,
                                          int batchSize,
                                          hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

template <typename T>
hipblasStatus_t cublasgetriBatched(hipblasHandle_t handle,
                                   int n,               // NOLINT
                                   const T* const A[],  // NOLINT
                                   int lda,
                                   const int* P,
                                   T* const C[],  // NOLINT
                                   int ldc,
                                   int* info,
                                   int batchSize,
                                   hipStream_t stream);

template <>
inline hipblasStatus_t cublasgetriBatched(  // NOLINT
  hipblasHandle_t handle,
  int n,
  const float* const A[],  // NOLINT
  int lda,
  const int* P,
  float* const C[],  // NOLINT
  int ldc,
  int* info,
  int batchSize,
  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
}

template <>
inline hipblasStatus_t cublasgetriBatched(  // NOLINT
  hipblasHandle_t handle,
  int n,
  const double* const A[],  // NOLINT
  int lda,
  const int* P,
  double* const C[],  // NOLINT
  int ldc,
  int* info,
  int batchSize,
  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
}

/** @} */

/**
 * @defgroup gelsbatched cublas gelsbatched calls
 * @{
 */

template <typename T>
inline hipblasStatus_t cublasgelsBatched(hipblasHandle_t handle,  // NOLINT
                                         hipblasOperation_t trans,
                                         int m,
                                         int n,
                                         int nrhs,
                                         T* Aarray[],  // NOLINT
                                         int lda,
                                         T* Carray[],  // NOLINT
                                         int ldc,
                                         int* info,
                                         int* devInfoArray,
                                         int batchSize,
                                         hipStream_t stream);

template <>
inline hipblasStatus_t cublasgelsBatched(hipblasHandle_t handle,  // NOLINT
                                         hipblasOperation_t trans,
                                         int m,
                                         int n,
                                         int nrhs,
                                         float* Aarray[],  // NOLINT
                                         int lda,
                                         float* Carray[],  // NOLINT
                                         int ldc,
                                         int* info,
                                         int* devInfoArray,
                                         int batchSize,
                                         hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSgelsBatched(
    handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

template <>
inline hipblasStatus_t cublasgelsBatched(hipblasHandle_t handle,  // NOLINT
                                         hipblasOperation_t trans,
                                         int m,
                                         int n,
                                         int nrhs,
                                         double* Aarray[],  // NOLINT
                                         int lda,
                                         double* Carray[],  // NOLINT
                                         int ldc,
                                         int* info,
                                         int* devInfoArray,
                                         int batchSize,
                                         hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDgelsBatched(
    handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

/** @} */

/**
 * @defgroup geam cublas geam calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgeam(hipblasHandle_t handle,
                           hipblasOperation_t transA,
                           hipblasOperation_t transB,
                           int m,
                           int n,
                           const T* alfa,
                           const T* A,
                           int lda,
                           const T* beta,
                           const T* B,
                           int ldb,
                           T* C,
                           int ldc,
                           hipStream_t stream);

template <>
inline hipblasStatus_t cublasgeam(hipblasHandle_t handle,
                                  hipblasOperation_t transA,
                                  hipblasOperation_t transB,
                                  int m,
                                  int n,
                                  const float* alfa,
                                  const float* A,
                                  int lda,
                                  const float* beta,
                                  const float* B,
                                  int ldb,
                                  float* C,
                                  int ldc,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSgeam(handle, transA, transB, m, n, alfa, A, lda, beta, B, ldb, C, ldc);
}

template <>
inline hipblasStatus_t cublasgeam(hipblasHandle_t handle,
                                  hipblasOperation_t transA,
                                  hipblasOperation_t transB,
                                  int m,
                                  int n,
                                  const double* alfa,
                                  const double* A,
                                  int lda,
                                  const double* beta,
                                  const double* B,
                                  int ldb,
                                  double* C,
                                  int ldc,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDgeam(handle, transA, transB, m, n, alfa, A, lda, beta, B, ldb, C, ldc);
}
/** @} */

/**
 * @defgroup symm cublas symm calls
 * @{
 */
template <typename T>
hipblasStatus_t cublassymm(hipblasHandle_t handle,
                           hipblasSideMode_t side,
                           hipblasFillMode_t uplo,
                           int m,
                           int n,
                           const T* alpha,
                           const T* A,
                           int lda,
                           const T* B,
                           int ldb,
                           const T* beta,
                           T* C,
                           int ldc,
                           hipStream_t stream);

template <>
inline hipblasStatus_t cublassymm(hipblasHandle_t handle,
                                  hipblasSideMode_t side,
                                  hipblasFillMode_t uplo,
                                  int m,
                                  int n,
                                  const float* alpha,
                                  const float* A,
                                  int lda,
                                  const float* B,
                                  int ldb,
                                  const float* beta,
                                  float* C,
                                  int ldc,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline hipblasStatus_t cublassymm(hipblasHandle_t handle,
                                  hipblasSideMode_t side,
                                  hipblasFillMode_t uplo,
                                  int m,
                                  int n,
                                  const double* alpha,
                                  const double* A,
                                  int lda,
                                  const double* B,
                                  int ldb,
                                  const double* beta,
                                  double* C,
                                  int ldc,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
/** @} */

/**
 * @defgroup syrk cublas syrk calls
 * @{
 */
template <typename T>
hipblasStatus_t cublassyrk(hipblasHandle_t handle,
                           hipblasFillMode_t uplo,
                           hipblasOperation_t trans,
                           int n,
                           int k,
                           const T* alpha,
                           const T* A,
                           int lda,
                           const T* beta,
                           T* C,
                           int ldc,
                           hipStream_t stream);

template <>
inline hipblasStatus_t cublassyrk(hipblasHandle_t handle,
                                  hipblasFillMode_t uplo,
                                  hipblasOperation_t trans,
                                  int n,
                                  int k,
                                  const float* alpha,
                                  const float* A,
                                  int lda,
                                  const float* beta,
                                  float* C,
                                  int ldc,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
inline hipblasStatus_t cublassyrk(hipblasHandle_t handle,
                                  hipblasFillMode_t uplo,
                                  hipblasOperation_t trans,
                                  int n,
                                  int k,
                                  const double* alpha,
                                  const double* A,
                                  int lda,
                                  const double* beta,
                                  double* C,
                                  int ldc,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
/** @} */

/**
 * @defgroup nrm2 cublas nrm2 calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasnrm2(
  hipblasHandle_t handle, int n, const T* x, int incx, T* result, hipStream_t stream);

template <>
inline hipblasStatus_t cublasnrm2(
  hipblasHandle_t handle, int n, const float* x, int incx, float* result, hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSnrm2(handle, n, x, incx, result);
}

template <>
inline hipblasStatus_t cublasnrm2(
  hipblasHandle_t handle, int n, const double* x, int incx, double* result, hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDnrm2(handle, n, x, incx, result);
}
/** @} */

template <typename T>
hipblasStatus_t cublastrsm(hipblasHandle_t handle,
                           hipblasSideMode_t side,
                           hipblasFillMode_t uplo,
                           hipblasOperation_t trans,
                           hipblasDiagType_t diag,
                           int m,
                           int n,
                           const T* alpha,
                           const T* A,
                           int lda,
                           T* B,
                           int ldb,
                           hipStream_t stream);

template <>
inline hipblasStatus_t cublastrsm(hipblasHandle_t handle,
                                  hipblasSideMode_t side,
                                  hipblasFillMode_t uplo,
                                  hipblasOperation_t trans,
                                  hipblasDiagType_t diag,
                                  int m,
                                  int n,
                                  const float* alpha,
                                  const float* A,
                                  int lda,
                                  float* B,
                                  int ldb,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
inline hipblasStatus_t cublastrsm(hipblasHandle_t handle,
                                  hipblasSideMode_t side,
                                  hipblasFillMode_t uplo,
                                  hipblasOperation_t trans,
                                  hipblasDiagType_t diag,
                                  int m,
                                  int n,
                                  const double* alpha,
                                  const double* A,
                                  int lda,
                                  double* B,
                                  int ldb,
                                  hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

/**
 * @defgroup dot cublas dot calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasdot(hipblasHandle_t handle,
                          int n,
                          const T* x,
                          int incx,
                          const T* y,
                          int incy,
                          T* result,
                          hipStream_t stream);

template <>
inline hipblasStatus_t cublasdot(hipblasHandle_t handle,
                                 int n,
                                 const float* x,
                                 int incx,
                                 const float* y,
                                 int incy,
                                 float* result,
                                 hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDotEx_v2(
    handle, n, x, HIP_R_32F, incx, y, HIP_R_32F, incy, result, HIP_R_32F, HIP_R_32F);
}

template <>
inline hipblasStatus_t cublasdot(hipblasHandle_t handle,
                                 int n,
                                 const double* x,
                                 int incx,
                                 const double* y,
                                 int incy,
                                 double* result,
                                 hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDotEx_v2(
    handle, n, x, HIP_R_64F, incx, y, HIP_R_64F, incy, result, HIP_R_64F, HIP_R_64F);
}
/** @} */

/**
 * @defgroup setpointermode cublas set pointer mode method
 * @{
 */
// no T dependency...
// template <typename T>
// hipblasStatus_t cublassetpointermode(  // NOLINT
//                                         hipblasHandle_t  handle,
//                                         hipblasPointerMode_t mode,
//                                         hipStream_t stream);

// template<>
inline hipblasStatus_t cublassetpointermode(hipblasHandle_t handle,
                                            hipblasPointerMode_t mode,
                                            hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSetPointerMode(handle, mode);
}
/** @} */

/**
 * @defgroup scal cublas dot calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasscal(
  hipblasHandle_t handle, int n, const T* alpha, T* x, int incx, hipStream_t stream);

template <>
inline hipblasStatus_t cublasscal(
  hipblasHandle_t handle, int n, const float* alpha, float* x, int incx, hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasSscal(handle, n, alpha, x, incx);
}

template <>
inline hipblasStatus_t cublasscal(
  hipblasHandle_t handle, int n, const double* alpha, double* x, int incx, hipStream_t stream)
{
  RAFT_CUBLAS_TRY(hipblasSetStream(handle, stream));
  return hipblasDscal(handle, n, alpha, x, incx);
}

/** @} */

}  // namespace detail
}  // namespace linalg
}  // namespace raft
