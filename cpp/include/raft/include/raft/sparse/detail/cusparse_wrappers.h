/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/core/cusparse_macros.hpp>
#include <raft/core/error.hpp>
#include <raft/linalg/transpose.cuh>

#include <rmm/device_uvector.hpp>

#include <hipsparse.h>

namespace raft {
namespace sparse {
namespace detail {

/**
 * @defgroup gather cusparse gather methods
 * @{
 */
inline hipsparseStatus_t cusparsegather(hipsparseHandle_t handle,
                                       hipsparseDnVecDescr_t vecY,
                                       hipsparseSpVecDescr_t vecX,
                                       hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseGather(handle, vecY, vecX);
}

template <
  typename T,
  typename std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>* = nullptr>
hipsparseStatus_t cusparsegthr(
  hipsparseHandle_t handle, int nnz, const T* vals, T* vals_sorted, int* d_P, hipStream_t stream)
{
  auto constexpr float_type = []() constexpr {
    if constexpr (std::is_same_v<T, float>) {
      return HIP_R_32F;
    } else if constexpr (std::is_same_v<T, double>) {
      return HIP_R_64F;
    }
  }();
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  auto dense_vector_descr  = hipsparseDnVecDescr_t{};
  auto sparse_vector_descr = hipsparseSpVecDescr_t{};
  CUSPARSE_CHECK(hipsparseCreateDnVec(
    &dense_vector_descr, nnz, static_cast<void*>(const_cast<T*>(vals)), float_type));
  CUSPARSE_CHECK(hipsparseCreateSpVec(&sparse_vector_descr,
                                     nnz,
                                     nnz,
                                     static_cast<void*>(d_P),
                                     static_cast<void*>(vals_sorted),
                                     HIPSPARSE_INDEX_32I,
                                     HIPSPARSE_INDEX_BASE_ZERO,
                                     float_type));
  auto return_value = hipsparseGather(handle, dense_vector_descr, sparse_vector_descr);
  CUSPARSE_CHECK(hipsparseDestroyDnVec(dense_vector_descr));
  CUSPARSE_CHECK(hipsparseDestroySpVec(sparse_vector_descr));
  return return_value;
}
/** @} */

/**
 * @defgroup coo2csr cusparse COO to CSR converter methods
 * @{
 */
template <typename T>
void cusparsecoo2csr(
  hipsparseHandle_t handle, const T* cooRowInd, int nnz, int m, T* csrRowPtr, hipStream_t stream);
template <>
inline void cusparsecoo2csr(hipsparseHandle_t handle,
                            const int* cooRowInd,
                            int nnz,
                            int m,
                            int* csrRowPtr,
                            hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  CUSPARSE_CHECK(hipsparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, HIPSPARSE_INDEX_BASE_ZERO));
}
/** @} */

/**
 * @defgroup coosort cusparse coo sort methods
 * @{
 */
template <typename T>
size_t cusparsecoosort_bufferSizeExt(  // NOLINT
  hipsparseHandle_t handle,
  int m,
  int n,
  int nnz,
  const T* cooRows,
  const T* cooCols,
  hipStream_t stream);
template <>
inline size_t cusparsecoosort_bufferSizeExt(  // NOLINT
  hipsparseHandle_t handle,
  int m,
  int n,
  int nnz,
  const int* cooRows,
  const int* cooCols,
  hipStream_t stream)
{
  size_t val;
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  CUSPARSE_CHECK(hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRows, cooCols, &val));
  return val;
}

template <typename T>
void cusparsecoosortByRow(  // NOLINT
  hipsparseHandle_t handle,
  int m,
  int n,
  int nnz,
  T* cooRows,
  T* cooCols,
  T* P,
  void* pBuffer,
  hipStream_t stream);
template <>
inline void cusparsecoosortByRow(  // NOLINT
  hipsparseHandle_t handle,
  int m,
  int n,
  int nnz,
  int* cooRows,
  int* cooCols,
  int* P,
  void* pBuffer,
  hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  CUSPARSE_CHECK(hipsparseXcoosortByRow(handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}
/** @} */

#if not defined CUDA_ENFORCE_LOWER and CUDA_VER_10_1_UP
/**
 * @defgroup cusparse Create CSR operations
 * @{
 */
template <typename ValueT, typename IndptrType, typename IndicesType>
hipsparseStatus_t cusparsecreatecsr(hipsparseSpMatDescr_t* spMatDescr,
                                   int64_t rows,
                                   int64_t cols,
                                   int64_t nnz,
                                   IndptrType* csrRowOffsets,
                                   IndicesType* csrColInd,
                                   ValueT* csrValues);
template <>
inline hipsparseStatus_t cusparsecreatecsr(hipsparseSpMatDescr_t* spMatDescr,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t nnz,
                                          int32_t* csrRowOffsets,
                                          int32_t* csrColInd,
                                          float* csrValues)
{
  return hipsparseCreateCsr(spMatDescr,
                           rows,
                           cols,
                           nnz,
                           csrRowOffsets,
                           csrColInd,
                           csrValues,
                           HIPSPARSE_INDEX_32I,
                           HIPSPARSE_INDEX_32I,
                           HIPSPARSE_INDEX_BASE_ZERO,
                           HIP_R_32F);
}
template <>
inline hipsparseStatus_t cusparsecreatecsr(hipsparseSpMatDescr_t* spMatDescr,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t nnz,
                                          int32_t* csrRowOffsets,
                                          int32_t* csrColInd,
                                          double* csrValues)
{
  return hipsparseCreateCsr(spMatDescr,
                           rows,
                           cols,
                           nnz,
                           csrRowOffsets,
                           csrColInd,
                           csrValues,
                           HIPSPARSE_INDEX_32I,
                           HIPSPARSE_INDEX_32I,
                           HIPSPARSE_INDEX_BASE_ZERO,
                           HIP_R_64F);
}
template <>
inline hipsparseStatus_t cusparsecreatecsr(hipsparseSpMatDescr_t* spMatDescr,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t nnz,
                                          int64_t* csrRowOffsets,
                                          int64_t* csrColInd,
                                          float* csrValues)
{
  return hipsparseCreateCsr(spMatDescr,
                           rows,
                           cols,
                           nnz,
                           csrRowOffsets,
                           csrColInd,
                           csrValues,
                           HIPSPARSE_INDEX_64I,
                           HIPSPARSE_INDEX_64I,
                           HIPSPARSE_INDEX_BASE_ZERO,
                           HIP_R_32F);
}
template <>
inline hipsparseStatus_t cusparsecreatecsr(hipsparseSpMatDescr_t* spMatDescr,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t nnz,
                                          int64_t* csrRowOffsets,
                                          int64_t* csrColInd,
                                          double* csrValues)
{
  return hipsparseCreateCsr(spMatDescr,
                           rows,
                           cols,
                           nnz,
                           csrRowOffsets,
                           csrColInd,
                           csrValues,
                           HIPSPARSE_INDEX_64I,
                           HIPSPARSE_INDEX_64I,
                           HIPSPARSE_INDEX_BASE_ZERO,
                           HIP_R_64F);
}
/** @} */
/**
 * @defgroup cusparse CreateDnVec operations
 * @{
 */
template <typename T>
hipsparseStatus_t cusparsecreatednvec(hipsparseDnVecDescr_t* dnVecDescr, int64_t size, T* values);
template <>
inline hipsparseStatus_t cusparsecreatednvec(hipsparseDnVecDescr_t* dnVecDescr,
                                            int64_t size,
                                            float* values)
{
  return hipsparseCreateDnVec(dnVecDescr, size, values, HIP_R_32F);
}
template <>
inline hipsparseStatus_t cusparsecreatednvec(hipsparseDnVecDescr_t* dnVecDescr,
                                            int64_t size,
                                            double* values)
{
  return hipsparseCreateDnVec(dnVecDescr, size, values, HIP_R_64F);
}
/** @} */

/**
 * @defgroup cusparse CreateDnMat operations
 * @{
 */
template <typename T>
hipsparseStatus_t cusparsecreatednmat(hipsparseDnMatDescr_t* dnMatDescr,
                                     int64_t rows,
                                     int64_t cols,
                                     int64_t ld,
                                     T* values,
                                     hipsparseOrder_t order);
template <>
inline hipsparseStatus_t cusparsecreatednmat(hipsparseDnMatDescr_t* dnMatDescr,
                                            int64_t rows,
                                            int64_t cols,
                                            int64_t ld,
                                            float* values,
                                            hipsparseOrder_t order)
{
  return hipsparseCreateDnMat(dnMatDescr, rows, cols, ld, values, HIP_R_32F, order);
}
template <>
inline hipsparseStatus_t cusparsecreatednmat(hipsparseDnMatDescr_t* dnMatDescr,
                                            int64_t rows,
                                            int64_t cols,
                                            int64_t ld,
                                            double* values,
                                            hipsparseOrder_t order)
{
  return hipsparseCreateDnMat(dnMatDescr, rows, cols, ld, values, HIP_R_64F, order);
}
/** @} */

/**
 * @defgroup Csrmv cusparse SpMV operations
 * @{
 */
template <typename T>
hipsparseStatus_t cusparsespmv_buffersize(hipsparseHandle_t handle,
                                         hipsparseOperation_t opA,
                                         const T* alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnVecDescr_t vecX,
                                         const T* beta,
                                         const hipsparseDnVecDescr_t vecY,
                                         hipsparseSpMVAlg_t alg,
                                         size_t* bufferSize,
                                         hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsespmv_buffersize(hipsparseHandle_t handle,
                                                hipsparseOperation_t opA,
                                                const float* alpha,
                                                const hipsparseSpMatDescr_t matA,
                                                const hipsparseDnVecDescr_t vecX,
                                                const float* beta,
                                                const hipsparseDnVecDescr_t vecY,
                                                hipsparseSpMVAlg_t alg,
                                                size_t* bufferSize,
                                                hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSpMV_bufferSize(
    handle, opA, alpha, matA, vecX, beta, vecY, HIP_R_32F, alg, bufferSize);
}
template <>
inline hipsparseStatus_t cusparsespmv_buffersize(hipsparseHandle_t handle,
                                                hipsparseOperation_t opA,
                                                const double* alpha,
                                                const hipsparseSpMatDescr_t matA,
                                                const hipsparseDnVecDescr_t vecX,
                                                const double* beta,
                                                const hipsparseDnVecDescr_t vecY,
                                                hipsparseSpMVAlg_t alg,
                                                size_t* bufferSize,
                                                hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSpMV_bufferSize(
    handle, opA, alpha, matA, vecX, beta, vecY, HIP_R_64F, alg, bufferSize);
}

template <typename T>
hipsparseStatus_t cusparsespmv(hipsparseHandle_t handle,
                              hipsparseOperation_t opA,
                              const T* alpha,
                              const hipsparseSpMatDescr_t matA,
                              const hipsparseDnVecDescr_t vecX,
                              const T* beta,
                              const hipsparseDnVecDescr_t vecY,
                              hipsparseSpMVAlg_t alg,
                              T* externalBuffer,
                              hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsespmv(hipsparseHandle_t handle,
                                     hipsparseOperation_t opA,
                                     const float* alpha,
                                     const hipsparseSpMatDescr_t matA,
                                     const hipsparseDnVecDescr_t vecX,
                                     const float* beta,
                                     const hipsparseDnVecDescr_t vecY,
                                     hipsparseSpMVAlg_t alg,
                                     float* externalBuffer,
                                     hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, HIP_R_32F, alg, externalBuffer);
}
template <>
inline hipsparseStatus_t cusparsespmv(hipsparseHandle_t handle,
                                     hipsparseOperation_t opA,
                                     const double* alpha,
                                     const hipsparseSpMatDescr_t matA,
                                     const hipsparseDnVecDescr_t vecX,
                                     const double* beta,
                                     const hipsparseDnVecDescr_t vecY,
                                     hipsparseSpMVAlg_t alg,
                                     double* externalBuffer,
                                     hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, HIP_R_64F, alg, externalBuffer);
}
/** @} */
#else
/**
 * @defgroup Csrmv cusparse csrmv operations
 * @{
 */
template <typename T>
hipsparseStatus_t cusparsecsrmv(  // NOLINT
  hipsparseHandle_t handle,
  hipsparseOperation_t trans,
  int m,
  int n,
  int nnz,
  const T* alpha,
  const hipsparseMatDescr_t descr,
  const T* csrVal,
  const int* csrRowPtr,
  const int* csrColInd,
  const T* x,
  const T* beta,
  T* y,
  hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsecsrmv(hipsparseHandle_t handle,
                                      hipsparseOperation_t trans,
                                      int m,
                                      int n,
                                      int nnz,
                                      const float* alpha,
                                      const hipsparseMatDescr_t descr,
                                      const float* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const float* x,
                                      const float* beta,
                                      float* y,
                                      hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseScsrmv(
    handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
}
template <>
inline hipsparseStatus_t cusparsecsrmv(hipsparseHandle_t handle,
                                      hipsparseOperation_t trans,
                                      int m,
                                      int n,
                                      int nnz,
                                      const double* alpha,
                                      const hipsparseMatDescr_t descr,
                                      const double* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const double* x,
                                      const double* beta,
                                      double* y,
                                      hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseDcsrmv(
    handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
}
/** @} */
#endif

#if not defined CUDA_ENFORCE_LOWER and CUDA_VER_10_1_UP
/**
 * @defgroup Csrmm cusparse csrmm operations
 * @{
 */
template <typename T>
hipsparseStatus_t cusparsespmm_bufferSize(hipsparseHandle_t handle,
                                         hipsparseOperation_t opA,
                                         hipsparseOperation_t opB,
                                         const T* alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnMatDescr_t matB,
                                         const T* beta,
                                         hipsparseDnMatDescr_t matC,
                                         hipsparseSpMMAlg_t alg,
                                         size_t* bufferSize,
                                         hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsespmm_bufferSize(hipsparseHandle_t handle,
                                                hipsparseOperation_t opA,
                                                hipsparseOperation_t opB,
                                                const float* alpha,
                                                const hipsparseSpMatDescr_t matA,
                                                const hipsparseDnMatDescr_t matB,
                                                const float* beta,
                                                hipsparseDnMatDescr_t matC,
                                                hipsparseSpMMAlg_t alg,
                                                size_t* bufferSize,
                                                hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSpMM_bufferSize(
    handle, opA, opB, alpha, matA, matB, beta, matC, HIP_R_32F, alg, bufferSize);
}
template <>
inline hipsparseStatus_t cusparsespmm_bufferSize(hipsparseHandle_t handle,
                                                hipsparseOperation_t opA,
                                                hipsparseOperation_t opB,
                                                const double* alpha,
                                                const hipsparseSpMatDescr_t matA,
                                                const hipsparseDnMatDescr_t matB,
                                                const double* beta,
                                                hipsparseDnMatDescr_t matC,
                                                hipsparseSpMMAlg_t alg,
                                                size_t* bufferSize,
                                                hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSpMM_bufferSize(
    handle, opA, opB, alpha, matA, matB, beta, matC, HIP_R_64F, alg, bufferSize);
}
template <typename T>
inline hipsparseStatus_t cusparsespmm(hipsparseHandle_t handle,
                                     hipsparseOperation_t opA,
                                     hipsparseOperation_t opB,
                                     const T* alpha,
                                     const hipsparseSpMatDescr_t matA,
                                     const hipsparseDnMatDescr_t matB,
                                     const T* beta,
                                     hipsparseDnMatDescr_t matC,
                                     hipsparseSpMMAlg_t alg,
                                     T* externalBuffer,
                                     hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsespmm(hipsparseHandle_t handle,
                                     hipsparseOperation_t opA,
                                     hipsparseOperation_t opB,
                                     const float* alpha,
                                     const hipsparseSpMatDescr_t matA,
                                     const hipsparseDnMatDescr_t matB,
                                     const float* beta,
                                     hipsparseDnMatDescr_t matC,
                                     hipsparseSpMMAlg_t alg,
                                     float* externalBuffer,
                                     hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSpMM(handle,
                      opA,
                      opB,
                      static_cast<void const*>(alpha),
                      matA,
                      matB,
                      static_cast<void const*>(beta),
                      matC,
                      HIP_R_32F,
                      alg,
                      static_cast<void*>(externalBuffer));
}
template <>
inline hipsparseStatus_t cusparsespmm(hipsparseHandle_t handle,
                                     hipsparseOperation_t opA,
                                     hipsparseOperation_t opB,
                                     const double* alpha,
                                     const hipsparseSpMatDescr_t matA,
                                     const hipsparseDnMatDescr_t matB,
                                     const double* beta,
                                     hipsparseDnMatDescr_t matC,
                                     hipsparseSpMMAlg_t alg,
                                     double* externalBuffer,
                                     hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSpMM(handle,
                      opA,
                      opB,
                      static_cast<void const*>(alpha),
                      matA,
                      matB,
                      static_cast<void const*>(beta),
                      matC,
                      HIP_R_64F,
                      alg,
                      static_cast<void*>(externalBuffer));
}

template <typename T>
hipsparseStatus_t cusparsesddmm_bufferSize(hipsparseHandle_t handle,
                                          hipsparseOperation_t opA,
                                          hipsparseOperation_t opB,
                                          const T* alpha,
                                          const hipsparseDnMatDescr_t matA,
                                          const hipsparseDnMatDescr_t matB,
                                          const T* beta,
                                          hipsparseSpMatDescr_t matC,
                                          hipsparseSDDMMAlg_t alg,
                                          size_t* bufferSize,
                                          hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsesddmm_bufferSize(hipsparseHandle_t handle,
                                                 hipsparseOperation_t opA,
                                                 hipsparseOperation_t opB,
                                                 const float* alpha,
                                                 const hipsparseDnMatDescr_t matA,
                                                 const hipsparseDnMatDescr_t matB,
                                                 const float* beta,
                                                 hipsparseSpMatDescr_t matC,
                                                 hipsparseSDDMMAlg_t alg,
                                                 size_t* bufferSize,
                                                 hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSDDMM_bufferSize(
    handle, opA, opB, alpha, matA, matB, beta, matC, HIP_R_32F, alg, bufferSize);
}
template <>
inline hipsparseStatus_t cusparsesddmm_bufferSize(hipsparseHandle_t handle,
                                                 hipsparseOperation_t opA,
                                                 hipsparseOperation_t opB,
                                                 const double* alpha,
                                                 const hipsparseDnMatDescr_t matA,
                                                 const hipsparseDnMatDescr_t matB,
                                                 const double* beta,
                                                 hipsparseSpMatDescr_t matC,
                                                 hipsparseSDDMMAlg_t alg,
                                                 size_t* bufferSize,
                                                 hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSDDMM_bufferSize(
    handle, opA, opB, alpha, matA, matB, beta, matC, HIP_R_64F, alg, bufferSize);
}
template <typename T>
inline hipsparseStatus_t cusparsesddmm(hipsparseHandle_t handle,
                                      hipsparseOperation_t opA,
                                      hipsparseOperation_t opB,
                                      const T* alpha,
                                      const hipsparseDnMatDescr_t matA,
                                      const hipsparseDnMatDescr_t matB,
                                      const T* beta,
                                      hipsparseSpMatDescr_t matC,
                                      hipsparseSDDMMAlg_t alg,
                                      T* externalBuffer,
                                      hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsesddmm(hipsparseHandle_t handle,
                                      hipsparseOperation_t opA,
                                      hipsparseOperation_t opB,
                                      const float* alpha,
                                      const hipsparseDnMatDescr_t matA,
                                      const hipsparseDnMatDescr_t matB,
                                      const float* beta,
                                      hipsparseSpMatDescr_t matC,
                                      hipsparseSDDMMAlg_t alg,
                                      float* externalBuffer,
                                      hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSDDMM(handle,
                       opA,
                       opB,
                       static_cast<void const*>(alpha),
                       matA,
                       matB,
                       static_cast<void const*>(beta),
                       matC,
                       HIP_R_32F,
                       alg,
                       static_cast<void*>(externalBuffer));
}
template <>
inline hipsparseStatus_t cusparsesddmm(hipsparseHandle_t handle,
                                      hipsparseOperation_t opA,
                                      hipsparseOperation_t opB,
                                      const double* alpha,
                                      const hipsparseDnMatDescr_t matA,
                                      const hipsparseDnMatDescr_t matB,
                                      const double* beta,
                                      hipsparseSpMatDescr_t matC,
                                      hipsparseSDDMMAlg_t alg,
                                      double* externalBuffer,
                                      hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSDDMM(handle,
                       opA,
                       opB,
                       static_cast<void const*>(alpha),
                       matA,
                       matB,
                       static_cast<void const*>(beta),
                       matC,
                       HIP_R_64F,
                       alg,
                       static_cast<void*>(externalBuffer));
}

/** @} */
#else
/**
 * @defgroup Csrmm cusparse csrmm operations
 * @{
 */
template <typename T>
hipsparseStatus_t cusparsecsrmm(  // NOLINT
  hipsparseHandle_t handle,
  hipsparseOperation_t trans,
  int m,
  int n,
  int k,
  int nnz,
  const T* alpha,
  const hipsparseMatDescr_t descr,
  const T* csrVal,
  const int* csrRowPtr,
  const int* csrColInd,
  const T* x,
  const int ldx,
  const T* beta,
  T* y,
  const int ldy,
  hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsecsrmm(hipsparseHandle_t handle,
                                      hipsparseOperation_t trans,
                                      int m,
                                      int n,
                                      int k,
                                      int nnz,
                                      const float* alpha,
                                      const hipsparseMatDescr_t descr,
                                      const float* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const float* x,
                                      const int ldx,
                                      const float* beta,
                                      float* y,
                                      const int ldy,
                                      hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseScsrmm(
    handle, trans, m, n, k, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, ldx, beta, y, ldy);
}
template <>
inline hipsparseStatus_t cusparsecsrmm(hipsparseHandle_t handle,
                                      hipsparseOperation_t trans,
                                      int m,
                                      int n,
                                      int k,
                                      int nnz,
                                      const double* alpha,
                                      const hipsparseMatDescr_t descr,
                                      const double* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const double* x,
                                      const int ldx,
                                      const double* beta,
                                      double* y,
                                      const int ldy,
                                      hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseDcsrmm(
    handle, trans, m, n, k, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, ldx, beta, y, ldy);
}
/** @} */
#endif

/**
 * @defgroup Gemmi cusparse gemmi operations
 * @{
 */
#if CUDART_VERSION < 12000
template <typename T>
hipsparseStatus_t cusparsegemmi(  // NOLINT
  hipsparseHandle_t handle,
  int m,
  int n,
  int k,
  int nnz,
  const T* alpha,
  const T* A,
  int lda,
  const T* cscValB,
  const int* cscColPtrB,
  const int* cscRowIndB,
  const T* beta,
  T* C,
  int ldc,
  hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsegemmi(hipsparseHandle_t handle,
                                      int m,
                                      int n,
                                      int k,
                                      int nnz,
                                      const float* alpha,
                                      const float* A,
                                      int lda,
                                      const float* cscValB,
                                      const int* cscColPtrB,
                                      const int* cscRowIndB,
                                      const float* beta,
                                      float* C,
                                      int ldc,
                                      hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return hipsparseSgemmi(
    handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
#pragma GCC diagnostic pop
}
template <>
inline hipsparseStatus_t cusparsegemmi(hipsparseHandle_t handle,
                                      int m,
                                      int n,
                                      int k,
                                      int nnz,
                                      const double* alpha,
                                      const double* A,
                                      int lda,
                                      const double* cscValB,
                                      const int* cscColPtrB,
                                      const int* cscRowIndB,
                                      const double* beta,
                                      double* C,
                                      int ldc,
                                      hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return hipsparseDgemmi(
    handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
#pragma GCC diagnostic pop
}
#else  // CUDART >= 12.0
template <typename T>
hipsparseStatus_t cusparsegemmi(  // NOLINT
  hipsparseHandle_t handle,
  int m,
  int n,
  int k,
  int nnz,
  const T* alpha,
  const T* A,
  int lda,
  const T* cscValB,
  const int* cscColPtrB,
  const int* cscRowIndB,
  const T* beta,
  T* C,
  int ldc,
  hipStream_t stream)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Unsupported data type");

  hipsparseDnMatDescr_t matA;
  hipsparseSpMatDescr_t matB;
  hipsparseDnMatDescr_t matC;
  rmm::device_uvector<T> CT(m * n, stream);

  auto constexpr math_type = std::is_same_v<T, float> ? HIP_R_32F : HIP_R_64F;
  // Create sparse matrix B
  CUSPARSE_CHECK(hipsparseCreateCsc(&matB,
                                   k,
                                   n,
                                   nnz,
                                   static_cast<void*>(const_cast<int*>(cscColPtrB)),
                                   static_cast<void*>(const_cast<int*>(cscRowIndB)),
                                   static_cast<void*>(const_cast<T*>(cscValB)),
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_BASE_ZERO,
                                   math_type));
  /**
   *  Create dense matrices.
   *  Note: Since this is replacing `cusparse_gemmi`, it assumes dense inputs are
   *  column-ordered
   */
  CUSPARSE_CHECK(hipsparseCreateDnMat(
    &matA, m, k, lda, static_cast<void*>(const_cast<T*>(A)), math_type, HIPSPARSE_ORDER_COL));
  CUSPARSE_CHECK(hipsparseCreateDnMat(
    &matC, n, m, n, static_cast<void*>(CT.data()), math_type, HIPSPARSE_ORDER_COL));

  auto opA         = HIPSPARSE_OPERATION_TRANSPOSE;
  auto opB         = HIPSPARSE_OPERATION_TRANSPOSE;
  auto alg         = HIPSPARSE_SPMM_CSR_ALG1;
  auto buffer_size = std::size_t{};

  CUSPARSE_CHECK(cusparsespmm_bufferSize(
    handle, opB, opA, alpha, matB, matA, beta, matC, alg, &buffer_size, stream));
  buffer_size = buffer_size / sizeof(T);
  rmm::device_uvector<T> external_buffer(buffer_size, stream);
  auto ext_buf = static_cast<T*>(static_cast<void*>(external_buffer.data()));
  auto return_value =
    cusparsespmm(handle, opB, opA, alpha, matB, matA, beta, matC, alg, ext_buf, stream);

  raft::resources rhandle;
  raft::linalg::transpose(rhandle, CT.data(), C, n, m, stream);
  // destroy matrix/vector descriptors
  CUSPARSE_CHECK(hipsparseDestroyDnMat(matA));
  CUSPARSE_CHECK(hipsparseDestroySpMat(matB));
  CUSPARSE_CHECK(hipsparseDestroyDnMat(matC));
  return return_value;
}
#endif
/** @} */

/**
 * @defgroup csr2coo cusparse CSR to COO converter methods
 * @{
 */
template <typename T>
void cusparsecsr2coo(  // NOLINT
  hipsparseHandle_t handle,
  const int n,
  const int nnz,
  const T* csrRowPtr,
  T* cooRowInd,
  hipStream_t stream);
template <>
inline void cusparsecsr2coo(hipsparseHandle_t handle,
                            const int n,
                            const int nnz,
                            const int* csrRowPtr,
                            int* cooRowInd,
                            hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  CUSPARSE_CHECK(hipsparseXcsr2coo(handle, csrRowPtr, nnz, n, cooRowInd, HIPSPARSE_INDEX_BASE_ZERO));
}
/** @} */

/**
 * @defgroup setpointermode cusparse set pointer mode method
 * @{
 */
// no T dependency...
// template <typename T>
// hipsparseStatus_t cusparsesetpointermode(  // NOLINT
//                                         hipsparseHandle_t handle,
//                                         hipsparsePointerMode_t mode,
//                                         hipStream_t stream);

// template<>
inline hipsparseStatus_t cusparsesetpointermode(hipsparseHandle_t handle,
                                               hipsparsePointerMode_t mode,
                                               hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSetPointerMode(handle, mode);
}
/** @} */

/**
 * @defgroup Csr2cscEx2 cusparse csr->csc conversion
 * @{
 */

template <typename T>
hipsparseStatus_t cusparsecsr2csc_bufferSize(hipsparseHandle_t handle,
                                            int m,
                                            int n,
                                            int nnz,
                                            const T* csrVal,
                                            const int* csrRowPtr,
                                            const int* csrColInd,
                                            void* cscVal,
                                            int* cscColPtr,
                                            int* cscRowInd,
                                            hipsparseAction_t copyValues,
                                            hipsparseIndexBase_t idxBase,
                                            hipsparseCsr2CscAlg_t alg,
                                            size_t* bufferSize,
                                            hipStream_t stream);

template <>
inline hipsparseStatus_t cusparsecsr2csc_bufferSize(hipsparseHandle_t handle,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   const float* csrVal,
                                                   const int* csrRowPtr,
                                                   const int* csrColInd,
                                                   void* cscVal,
                                                   int* cscColPtr,
                                                   int* cscRowInd,
                                                   hipsparseAction_t copyValues,
                                                   hipsparseIndexBase_t idxBase,
                                                   hipsparseCsr2CscAlg_t alg,
                                                   size_t* bufferSize,
                                                   hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));

  return hipsparseCsr2cscEx2_bufferSize(handle,
                                       m,
                                       n,
                                       nnz,
                                       csrVal,
                                       csrRowPtr,
                                       csrColInd,
                                       cscVal,
                                       cscColPtr,
                                       cscRowInd,
                                       HIP_R_32F,
                                       copyValues,
                                       idxBase,
                                       alg,
                                       bufferSize);
}
template <>
inline hipsparseStatus_t cusparsecsr2csc_bufferSize(hipsparseHandle_t handle,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   const double* csrVal,
                                                   const int* csrRowPtr,
                                                   const int* csrColInd,
                                                   void* cscVal,
                                                   int* cscColPtr,
                                                   int* cscRowInd,
                                                   hipsparseAction_t copyValues,
                                                   hipsparseIndexBase_t idxBase,
                                                   hipsparseCsr2CscAlg_t alg,
                                                   size_t* bufferSize,
                                                   hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));

  return hipsparseCsr2cscEx2_bufferSize(handle,
                                       m,
                                       n,
                                       nnz,
                                       csrVal,
                                       csrRowPtr,
                                       csrColInd,
                                       cscVal,
                                       cscColPtr,
                                       cscRowInd,
                                       HIP_R_64F,
                                       copyValues,
                                       idxBase,
                                       alg,
                                       bufferSize);
}

template <typename T>
hipsparseStatus_t cusparsecsr2csc(hipsparseHandle_t handle,
                                 int m,
                                 int n,
                                 int nnz,
                                 const T* csrVal,
                                 const int* csrRowPtr,
                                 const int* csrColInd,
                                 void* cscVal,
                                 int* cscColPtr,
                                 int* cscRowInd,
                                 hipsparseAction_t copyValues,
                                 hipsparseIndexBase_t idxBase,
                                 hipsparseCsr2CscAlg_t alg,
                                 void* buffer,
                                 hipStream_t stream);

template <>
inline hipsparseStatus_t cusparsecsr2csc(hipsparseHandle_t handle,
                                        int m,
                                        int n,
                                        int nnz,
                                        const float* csrVal,
                                        const int* csrRowPtr,
                                        const int* csrColInd,
                                        void* cscVal,
                                        int* cscColPtr,
                                        int* cscRowInd,
                                        hipsparseAction_t copyValues,
                                        hipsparseIndexBase_t idxBase,
                                        hipsparseCsr2CscAlg_t alg,
                                        void* buffer,
                                        hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));

  return hipsparseCsr2cscEx2(handle,
                            m,
                            n,
                            nnz,
                            csrVal,
                            csrRowPtr,
                            csrColInd,
                            cscVal,
                            cscColPtr,
                            cscRowInd,
                            HIP_R_32F,
                            copyValues,
                            idxBase,
                            alg,
                            buffer);
}

template <>
inline hipsparseStatus_t cusparsecsr2csc(hipsparseHandle_t handle,
                                        int m,
                                        int n,
                                        int nnz,
                                        const double* csrVal,
                                        const int* csrRowPtr,
                                        const int* csrColInd,
                                        void* cscVal,
                                        int* cscColPtr,
                                        int* cscRowInd,
                                        hipsparseAction_t copyValues,
                                        hipsparseIndexBase_t idxBase,
                                        hipsparseCsr2CscAlg_t alg,
                                        void* buffer,
                                        hipStream_t stream)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));

  return hipsparseCsr2cscEx2(handle,
                            m,
                            n,
                            nnz,
                            csrVal,
                            csrRowPtr,
                            csrColInd,
                            cscVal,
                            cscColPtr,
                            cscRowInd,
                            HIP_R_64F,
                            copyValues,
                            idxBase,
                            alg,
                            buffer);
}

/** @} */

/**
 * @defgroup csrgemm2 cusparse sparse gemm operations
 * @{
 */

template <typename T>
hipsparseStatus_t cusparsecsr2dense_buffersize(hipsparseHandle_t handle,
                                              int m,
                                              int n,
                                              int nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const T* csrValA,
                                              const int* csrRowPtrA,
                                              const int* csrColIndA,
                                              T* A,
                                              int lda,
                                              size_t* buffer_size,
                                              hipStream_t stream,
                                              bool row_major = false);

template <>
inline hipsparseStatus_t cusparsecsr2dense_buffersize(hipsparseHandle_t handle,
                                                     int m,
                                                     int n,
                                                     int nnz,
                                                     const hipsparseMatDescr_t descrA,
                                                     const float* csrValA,
                                                     const int* csrRowPtrA,
                                                     const int* csrColIndA,
                                                     float* A,
                                                     int lda,
                                                     size_t* buffer_size,
                                                     hipStream_t stream,
                                                     bool row_major)
{
#if CUDART_VERSION >= 11020
  hipsparseOrder_t order = row_major ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;

  hipsparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    static_cast<int64_t>(m),
                    static_cast<int64_t>(n),
                    static_cast<int64_t>(nnz),
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<float*>(csrValA));

  hipsparseDnMatDescr_t matB;
  cusparsecreatednmat(&matB,
                      static_cast<int64_t>(m),
                      static_cast<int64_t>(n),
                      static_cast<int64_t>(lda),
                      const_cast<float*>(A),
                      order);

  hipsparseStatus_t result = hipsparseSparseToDense_bufferSize(
    handle, matA, matB, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer_size);

  RAFT_CUSPARSE_TRY_NO_THROW(hipsparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(hipsparseDestroyDnMat(matB));

#else

  hipsparseStatus_t result = HIPSPARSE_STATUS_SUCCESS;
  buffer_size[0]          = 0;

#endif
  return result;
}

template <>
inline hipsparseStatus_t cusparsecsr2dense_buffersize(hipsparseHandle_t handle,
                                                     int m,
                                                     int n,
                                                     int nnz,
                                                     const hipsparseMatDescr_t descrA,
                                                     const double* csrValA,
                                                     const int* csrRowPtrA,
                                                     const int* csrColIndA,
                                                     double* A,
                                                     int lda,
                                                     size_t* buffer_size,
                                                     hipStream_t stream,
                                                     bool row_major)
{
#if CUDART_VERSION >= 11020
  hipsparseOrder_t order = row_major ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;
  hipsparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    static_cast<int64_t>(m),
                    static_cast<int64_t>(n),
                    static_cast<int64_t>(nnz),
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<double*>(csrValA));

  hipsparseDnMatDescr_t matB;
  cusparsecreatednmat(&matB,
                      static_cast<int64_t>(m),
                      static_cast<int64_t>(n),
                      static_cast<int64_t>(lda),
                      const_cast<double*>(A),
                      order);

  hipsparseStatus_t result = hipsparseSparseToDense_bufferSize(
    handle, matA, matB, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer_size);

  RAFT_CUSPARSE_TRY_NO_THROW(hipsparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(hipsparseDestroyDnMat(matB));

#else
  hipsparseStatus_t result = HIPSPARSE_STATUS_SUCCESS;
  buffer_size[0]          = 0;

#endif

  return result;
}

template <typename T>
hipsparseStatus_t cusparsecsr2dense(hipsparseHandle_t handle,
                                   int m,
                                   int n,
                                   int nnz,
                                   const hipsparseMatDescr_t descrA,
                                   const T* csrValA,
                                   const int* csrRowPtrA,
                                   const int* csrColIndA,
                                   T* A,
                                   int lda,
                                   void* buffer,
                                   hipStream_t stream,
                                   bool row_major = false);

template <>
inline hipsparseStatus_t cusparsecsr2dense(hipsparseHandle_t handle,
                                          int m,
                                          int n,
                                          int nnz,
                                          const hipsparseMatDescr_t descrA,
                                          const float* csrValA,
                                          const int* csrRowPtrA,
                                          const int* csrColIndA,
                                          float* A,
                                          int lda,
                                          void* buffer,
                                          hipStream_t stream,
                                          bool row_major)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));

#if CUDART_VERSION >= 11020
  hipsparseOrder_t order = row_major ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;
  hipsparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    static_cast<int64_t>(m),
                    static_cast<int64_t>(n),
                    static_cast<int64_t>(nnz),
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<float*>(csrValA));

  hipsparseDnMatDescr_t matB;
  cusparsecreatednmat(&matB,
                      static_cast<int64_t>(m),
                      static_cast<int64_t>(n),
                      static_cast<int64_t>(lda),
                      const_cast<float*>(A),
                      order);

  hipsparseStatus_t result =
    hipsparseSparseToDense(handle, matA, matB, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer);

  RAFT_CUSPARSE_TRY_NO_THROW(hipsparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(hipsparseDestroyDnMat(matB));

  return result;
#else
  return hipsparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
#endif
}
template <>
inline hipsparseStatus_t cusparsecsr2dense(hipsparseHandle_t handle,
                                          int m,
                                          int n,
                                          int nnz,
                                          const hipsparseMatDescr_t descrA,
                                          const double* csrValA,
                                          const int* csrRowPtrA,
                                          const int* csrColIndA,
                                          double* A,
                                          int lda,
                                          void* buffer,
                                          hipStream_t stream,
                                          bool row_major)
{
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));

#if CUDART_VERSION >= 11020
  hipsparseOrder_t order = row_major ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;
  hipsparseSpMatDescr_t matA;
  cusparsecreatecsr(&matA,
                    static_cast<int64_t>(m),
                    static_cast<int64_t>(n),
                    static_cast<int64_t>(nnz),
                    const_cast<int*>(csrRowPtrA),
                    const_cast<int*>(csrColIndA),
                    const_cast<double*>(csrValA));

  hipsparseDnMatDescr_t matB;
  cusparsecreatednmat(&matB,
                      static_cast<int64_t>(m),
                      static_cast<int64_t>(n),
                      static_cast<int64_t>(lda),
                      const_cast<double*>(A),
                      order);

  hipsparseStatus_t result =
    hipsparseSparseToDense(handle, matA, matB, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer);

  RAFT_CUSPARSE_TRY_NO_THROW(hipsparseDestroySpMat(matA));
  RAFT_CUSPARSE_TRY_NO_THROW(hipsparseDestroyDnMat(matB));

  return result;
#else

  return hipsparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
#endif
}

/** @} */

}  // namespace detail
}  // namespace sparse
}  // namespace raft
