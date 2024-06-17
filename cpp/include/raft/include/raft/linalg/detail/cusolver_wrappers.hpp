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

#include <raft/core/cusolver_macros.hpp>
#include <raft/util/cudart_utils.hpp>

#include <hipsolver.h>

#include <type_traits>

namespace raft {
namespace linalg {
namespace detail {

/**
 * @defgroup Getrf cusolver getrf operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDngetrf(hipsolverHandle_t handle,
                                  int m,  // NOLINT
                                  int n,
                                  T* A,
                                  int lda,
                                  T* Workspace,
                                  int* devIpiv,
                                  int* devInfo,
                                  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDngetrf(hipsolverHandle_t handle,  // NOLINT
                                         int m,
                                         int n,
                                         float* A,
                                         int lda,
                                         float* Workspace,
                                         int* devIpiv,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
inline hipsolverStatus_t cusolverDngetrf(hipsolverHandle_t handle,  // NOLINT
                                         int m,
                                         int n,
                                         double* A,
                                         int lda,
                                         double* Workspace,
                                         int* devIpiv,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <typename T>
hipsolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  T* A,
  int lda,
  int* Lwork);

template <>
inline hipsolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return hipsolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

template <>
inline hipsolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return hipsolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

/**
 * @defgroup Getrs cusolver getrs operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDngetrs(hipsolverHandle_t handle,  // NOLINT
                                  hipblasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const T* A,
                                  int lda,
                                  const int* devIpiv,
                                  T* B,
                                  int ldb,
                                  int* devInfo,
                                  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDngetrs(hipsolverHandle_t handle,  // NOLINT
                                         hipblasOperation_t trans,
                                         int n,
                                         int nrhs,
                                         const float* A,
                                         int lda,
                                         const int* devIpiv,
                                         float* B,
                                         int ldb,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

template <>
inline hipsolverStatus_t cusolverDngetrs(hipsolverHandle_t handle,  // NOLINT
                                         hipblasOperation_t trans,
                                         int n,
                                         int nrhs,
                                         const double* A,
                                         int lda,
                                         const int* devIpiv,
                                         double* B,
                                         int ldb,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}
/** @} */

/**
 * @defgroup syevd cusolver syevd operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  const T* W,
  int* lwork);

template <>
inline hipsolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  const float* W,
  int* lwork)
{
  return hipsolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

template <>
inline hipsolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  const double* W,
  int* lwork)
{
  return hipsolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}
/** @} */

/**
 * @defgroup syevj cusolver syevj operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnsyevj(hipsolverHandle_t handle,  // NOLINT
                                  hipsolverEigMode_t jobz,
                                  hipblasFillMode_t uplo,
                                  int n,
                                  T* A,
                                  int lda,
                                  T* W,
                                  T* work,
                                  int lwork,
                                  int* info,
                                  hipsolverSyevjInfo_t params,
                                  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDnsyevj(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  float* W,
  float* work,
  int lwork,
  int* info,
  hipsolverSyevjInfo_t params,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

template <>
inline hipsolverStatus_t cusolverDnsyevj(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  double* W,
  double* work,
  int lwork,
  int* info,
  hipsolverSyevjInfo_t params,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

template <typename T>
hipsolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  const T* W,
  int* lwork,
  hipsolverSyevjInfo_t params);

template <>
inline hipsolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  const float* W,
  int* lwork,
  hipsolverSyevjInfo_t params)
{
  return hipsolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

template <>
inline hipsolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  const double* W,
  int* lwork,
  hipsolverSyevjInfo_t params)
{
  return hipsolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
}
/** @} */

/**
 * @defgroup syevd cusolver syevd operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnsyevd(hipsolverHandle_t handle,  // NOLINT
                                  hipsolverEigMode_t jobz,
                                  hipblasFillMode_t uplo,
                                  int n,
                                  T* A,
                                  int lda,
                                  T* W,
                                  T* work,
                                  int lwork,
                                  int* devInfo,
                                  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDnsyevd(hipsolverHandle_t handle,  // NOLINT
                                         hipsolverEigMode_t jobz,
                                         hipblasFillMode_t uplo,
                                         int n,
                                         float* A,
                                         int lda,
                                         float* W,
                                         float* work,
                                         int lwork,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}

template <>
inline hipsolverStatus_t cusolverDnsyevd(hipsolverHandle_t handle,  // NOLINT
                                         hipsolverEigMode_t jobz,
                                         hipblasFillMode_t uplo,
                                         int n,
                                         double* A,
                                         int lda,
                                         double* W,
                                         double* work,
                                         int lwork,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}
/** @} */

/**
 * @defgroup syevdx cusolver syevdx operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipsolverEigRange_t range,
  hipblasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  T vl,
  T vu,
  int il,
  int iu,
  int* h_meig,
  const T* W,
  int* lwork);

template <>
inline hipsolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipsolverEigRange_t range,
  hipblasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  float vl,
  float vu,
  int il,
  int iu,
  int* h_meig,
  const float* W,
  int* lwork)
{
  return hipsolverDnSsyevdx_bufferSize(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, lwork);
}

template <>
inline hipsolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipsolverEigRange_t range,
  hipblasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  double vl,
  double vu,
  int il,
  int iu,
  int* h_meig,
  const double* W,
  int* lwork)
{
  return hipsolverDnDsyevdx_bufferSize(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, lwork);
}

template <typename T>
hipsolverStatus_t cusolverDnsyevdx(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipsolverEigRange_t range,
  hipblasFillMode_t uplo,
  int n,
  T* A,
  int lda,
  T vl,
  T vu,
  int il,
  int iu,
  int* h_meig,
  T* W,
  T* work,
  int lwork,
  int* devInfo,
  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDnsyevdx(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipsolverEigRange_t range,
  hipblasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  float vl,
  float vu,
  int il,
  int iu,
  int* h_meig,
  float* W,
  float* work,
  int lwork,
  int* devInfo,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSsyevdx(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, work, lwork, devInfo);
}

template <>
inline hipsolverStatus_t cusolverDnsyevdx(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  hipsolverEigRange_t range,
  hipblasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  double vl,
  double vu,
  int il,
  int iu,
  int* h_meig,
  double* W,
  double* work,
  int lwork,
  int* devInfo,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDsyevdx(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, work, lwork, devInfo);
}
/** @} */

/**
 * @defgroup svd cusolver svd operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDngesvd_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  int* lwork)
{
  if (std::is_same<std::decay_t<T>, float>::value) {
    return hipsolverDnSgesvd_bufferSize(handle, m, n, lwork);
  } else {
    return hipsolverDnDgesvd_bufferSize(handle, m, n, lwork);
  }
}
template <typename T>
hipsolverStatus_t cusolverDngesvd(  // NOLINT
  hipsolverHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  T* A,
  int lda,
  T* S,
  T* U,
  int ldu,
  T* VT,
  int ldvt,
  T* work,
  int lwork,
  T* rwork,
  int* devInfo,
  hipStream_t stream);
template <>
inline hipsolverStatus_t cusolverDngesvd(  // NOLINT
  hipsolverHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  float* A,
  int lda,
  float* S,
  float* U,
  int ldu,
  float* VT,
  int ldvt,
  float* work,
  int lwork,
  float* rwork,
  int* devInfo,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSgesvd(
    handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}
template <>
inline hipsolverStatus_t cusolverDngesvd(  // NOLINT
  hipsolverHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  double* A,
  int lda,
  double* S,
  double* U,
  int ldu,
  double* VT,
  int ldvt,
  double* work,
  int lwork,
  double* rwork,
  int* devInfo,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDgesvd(
    handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}

template <typename T>
inline hipsolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const T* A,
  int lda,
  const T* S,
  const T* U,
  int ldu,
  const T* V,
  int ldv,
  int* lwork,
  hipsolverGesvdjInfo_t params);
template <>
inline hipsolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const float* A,
  int lda,
  const float* S,
  const float* U,
  int ldu,
  const float* V,
  int ldv,
  int* lwork,
  hipsolverGesvdjInfo_t params)
{
  return hipsolverDnSgesvdj_bufferSize(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}
template <>
inline hipsolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const double* A,
  int lda,
  const double* S,
  const double* U,
  int ldu,
  const double* V,
  int ldv,
  int* lwork,
  hipsolverGesvdjInfo_t params)
{
  return hipsolverDnDgesvdj_bufferSize(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}
template <typename T>
inline hipsolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  T* A,
  int lda,
  T* S,
  T* U,
  int ldu,
  T* V,
  int ldv,
  T* work,
  int lwork,
  int* info,
  hipsolverGesvdjInfo_t params,
  hipStream_t stream);
template <>
inline hipsolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  float* A,
  int lda,
  float* S,
  float* U,
  int ldu,
  float* V,
  int ldv,
  float* work,
  int lwork,
  int* info,
  hipsolverGesvdjInfo_t params,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}
template <>
inline hipsolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  hipsolverHandle_t handle,
  hipsolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  double* A,
  int lda,
  double* S,
  double* U,
  int ldu,
  double* V,
  int ldv,
  double* work,
  int lwork,
  int* info,
  hipsolverGesvdjInfo_t params,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}

#if CUDART_VERSION >= 11010
template <typename T>
hipsolverStatus_t cusolverDnxgesvdr_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  signed char jobu,
  signed char jobv,
  int64_t m,
  int64_t n,
  int64_t k,
  int64_t p,
  int64_t niters,
  const T* a,
  int64_t lda,
  const T* Srand,
  const T* Urand,
  int64_t ldUrand,
  const T* Vrand,
  int64_t ldVrand,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  hipStream_t stream)
{
  RAFT_EXPECTS(std::is_floating_point_v<T>, "Unsupported data type");
  hipDataType dataType = std::is_same_v<T, float> ? HIP_R_32F : HIP_R_64F;
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  cusolverDnParams_t dn_params = nullptr;
  RAFT_CUSOLVER_TRY(cusolverDnCreateParams(&dn_params));
  auto result = cusolverDnXgesvdr_bufferSize(handle,
                                             dn_params,
                                             jobu,
                                             jobv,
                                             m,
                                             n,
                                             k,
                                             p,
                                             niters,
                                             dataType,
                                             a,
                                             lda,
                                             dataType,
                                             Srand,
                                             dataType,
                                             Urand,
                                             ldUrand,
                                             dataType,
                                             Vrand,
                                             ldVrand,
                                             dataType,
                                             workspaceInBytesOnDevice,
                                             workspaceInBytesOnHost);
  RAFT_CUSOLVER_TRY(cusolverDnDestroyParams(dn_params));
  return result;
}
template <typename T>
hipsolverStatus_t cusolverDnxgesvdr(  // NOLINT
  hipsolverHandle_t handle,
  signed char jobu,
  signed char jobv,
  int64_t m,
  int64_t n,
  int64_t k,
  int64_t p,
  int64_t niters,
  T* a,
  int64_t lda,
  T* Srand,
  T* Urand,
  int64_t ldUrand,
  T* Vrand,
  int64_t ldVrand,
  void* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  void* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* d_info,
  hipStream_t stream)
{
  hipDataType dataType = std::is_same_v<T, float> ? HIP_R_32F : HIP_R_64F;
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  cusolverDnParams_t dn_params = nullptr;
  RAFT_CUSOLVER_TRY(cusolverDnCreateParams(&dn_params));
  auto result = cusolverDnXgesvdr(handle,
                                  dn_params,
                                  jobu,
                                  jobv,
                                  m,
                                  n,
                                  k,
                                  p,
                                  niters,
                                  dataType,
                                  a,
                                  lda,
                                  dataType,
                                  Srand,
                                  dataType,
                                  Urand,
                                  ldUrand,
                                  dataType,
                                  Vrand,
                                  ldVrand,
                                  dataType,
                                  bufferOnDevice,
                                  workspaceInBytesOnDevice,
                                  bufferOnHost,
                                  workspaceInBytesOnHost,
                                  d_info);
  RAFT_CUSOLVER_TRY(cusolverDnDestroyParams(dn_params));
  return result;
}
#endif  // CUDART_VERSION >= 11010

/** @} */

/**
 * @defgroup potrf cusolver potrf operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipblasFillMode_t uplo,
  int n,
  T* A,
  int lda,
  int* Lwork);

template <>
inline hipsolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipblasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return hipsolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

template <>
inline hipsolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipblasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return hipsolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

template <typename T>
inline hipsolverStatus_t cusolverDnpotrf(hipsolverHandle_t handle,  // NOLINT
                                         hipblasFillMode_t uplo,
                                         int n,
                                         T* A,
                                         int lda,
                                         T* Workspace,
                                         int Lwork,
                                         int* devInfo,
                                         hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDnpotrf(hipsolverHandle_t handle,  // NOLINT
                                         hipblasFillMode_t uplo,
                                         int n,
                                         float* A,
                                         int lda,
                                         float* Workspace,
                                         int Lwork,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

template <>
inline hipsolverStatus_t cusolverDnpotrf(hipsolverHandle_t handle,  // NOLINT
                                         hipblasFillMode_t uplo,
                                         int n,
                                         double* A,
                                         int lda,
                                         double* Workspace,
                                         int Lwork,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}
/** @} */

/**
 * @defgroup potrs cusolver potrs operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnpotrs(hipsolverHandle_t handle,  // NOLINT
                                  hipblasFillMode_t uplo,
                                  int n,
                                  int nrhs,
                                  const T* A,
                                  int lda,
                                  T* B,
                                  int ldb,
                                  int* devInfo,
                                  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDnpotrs(hipsolverHandle_t handle,  // NOLINT
                                         hipblasFillMode_t uplo,
                                         int n,
                                         int nrhs,
                                         const float* A,
                                         int lda,
                                         float* B,
                                         int ldb,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

template <>
inline hipsolverStatus_t cusolverDnpotrs(hipsolverHandle_t handle,  // NOLINT
                                         hipblasFillMode_t uplo,
                                         int n,
                                         int nrhs,
                                         const double* A,
                                         int lda,
                                         double* B,
                                         int ldb,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}
/** @} */

/**
 * @defgroup geqrf cusolver geqrf operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDngeqrf(hipsolverHandle_t handle,
                                  int m,  // NOLINT
                                  int n,
                                  T* A,
                                  int lda,
                                  T* TAU,
                                  T* Workspace,
                                  int Lwork,
                                  int* devInfo,
                                  hipStream_t stream);
template <>
inline hipsolverStatus_t cusolverDngeqrf(hipsolverHandle_t handle,  // NOLINT
                                         int m,
                                         int n,
                                         float* A,
                                         int lda,
                                         float* TAU,
                                         float* Workspace,
                                         int Lwork,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}
template <>
inline hipsolverStatus_t cusolverDngeqrf(hipsolverHandle_t handle,  // NOLINT
                                         int m,
                                         int n,
                                         double* A,
                                         int lda,
                                         double* TAU,
                                         double* Workspace,
                                         int Lwork,
                                         int* devInfo,
                                         hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

template <typename T>
hipsolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  T* A,
  int lda,
  int* Lwork);
template <>
inline hipsolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return hipsolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
template <>
inline hipsolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return hipsolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
/** @} */

/**
 * @defgroup orgqr cusolver orgqr operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnorgqr(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  int k,
  T* A,
  int lda,
  const T* tau,
  T* work,
  int lwork,
  int* devInfo,
  hipStream_t stream);
template <>
inline hipsolverStatus_t cusolverDnorgqr(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  int k,
  float* A,
  int lda,
  const float* tau,
  float* work,
  int lwork,
  int* devInfo,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}
template <>
inline hipsolverStatus_t cusolverDnorgqr(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  int k,
  double* A,
  int lda,
  const double* tau,
  double* work,
  int lwork,
  int* devInfo,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

template <typename T>
hipsolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  int k,
  const T* A,
  int lda,
  const T* TAU,
  int* lwork);
template <>
inline hipsolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* TAU,
  int* lwork)
{
  return hipsolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, TAU, lwork);
}
template <>
inline hipsolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* TAU,
  int* lwork)
{
  return hipsolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, TAU, lwork);
}
/** @} */

/**
 * @defgroup ormqr cusolver ormqr operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnormqr(hipsolverHandle_t handle,  // NOLINT
                                  hipblasSideMode_t side,
                                  hipblasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const T* A,
                                  int lda,
                                  const T* tau,
                                  T* C,
                                  int ldc,
                                  T* work,
                                  int lwork,
                                  int* devInfo,
                                  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDnormqr(  // NOLINT
  hipsolverHandle_t handle,
  hipblasSideMode_t side,
  hipblasOperation_t trans,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* tau,
  float* C,
  int ldc,
  float* work,
  int lwork,
  int* devInfo,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

template <>
inline hipsolverStatus_t cusolverDnormqr(  // NOLINT
  hipsolverHandle_t handle,
  hipblasSideMode_t side,
  hipblasOperation_t trans,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* tau,
  double* C,
  int ldc,
  double* work,
  int lwork,
  int* devInfo,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return hipsolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

template <typename T>
hipsolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipblasSideMode_t side,
  hipblasOperation_t trans,
  int m,
  int n,
  int k,
  const T* A,
  int lda,
  const T* tau,
  const T* C,
  int ldc,
  int* lwork);

template <>
inline hipsolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipblasSideMode_t side,
  hipblasOperation_t trans,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* tau,
  const float* C,
  int ldc,
  int* lwork)
{
  return hipsolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

template <>
inline hipsolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  hipblasSideMode_t side,
  hipblasOperation_t trans,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* tau,
  const double* C,
  int ldc,
  int* lwork)
{
  return hipsolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}
/** @} */

/**
 * @defgroup csrqrBatched cusolver batched
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  hipsolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const hipsparseMatDescr_t descrA,
  const T* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes);

template <>
inline hipsolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  hipsolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const hipsparseMatDescr_t descrA,
  const float* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes)
{
  return cusolverSpScsrqrBufferInfoBatched(handle,
                                           m,
                                           n,
                                           nnzA,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           batchSize,
                                           info,
                                           internalDataInBytes,
                                           workspaceInBytes);
}

template <>
inline hipsolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  hipsolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const hipsparseMatDescr_t descrA,
  const double* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes)
{
  return cusolverSpDcsrqrBufferInfoBatched(handle,
                                           m,
                                           n,
                                           nnzA,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           batchSize,
                                           info,
                                           internalDataInBytes,
                                           workspaceInBytes);
}

template <typename T>
hipsolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  hipsolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const hipsparseMatDescr_t descrA,
  const T* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const T* b,
  T* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  hipsolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const hipsparseMatDescr_t descrA,
  const float* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const float* b,
  float* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverSpSetStream(handle, stream));
  return cusolverSpScsrqrsvBatched(
    handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}

template <>
inline hipsolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  hipsolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const hipsparseMatDescr_t descrA,
  const double* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const double* b,
  double* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverSpSetStream(handle, stream));
  return cusolverSpDcsrqrsvBatched(
    handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}
/** @} */

#if CUDART_VERSION >= 11010
/**
 * @defgroup DnXsyevd cusolver DnXsyevd operations
 * @{
 */
template <typename T>
hipsolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  cusolverDnParams_t params,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int64_t n,
  const T* A,
  int64_t lda,
  const T* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  cusolverDnParams_t params,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int64_t n,
  const float* A,
  int64_t lda,
  const float* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return cusolverDnXsyevd_bufferSize(handle,
                                     params,
                                     jobz,
                                     uplo,
                                     n,
                                     HIP_R_32F,
                                     A,
                                     lda,
                                     HIP_R_32F,
                                     W,
                                     HIP_R_32F,
                                     workspaceInBytesOnDevice,
                                     workspaceInBytesOnHost);
}

template <>
inline hipsolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  hipsolverHandle_t handle,
  cusolverDnParams_t params,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int64_t n,
  const double* A,
  int64_t lda,
  const double* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return cusolverDnXsyevd_bufferSize(handle,
                                     params,
                                     jobz,
                                     uplo,
                                     n,
                                     HIP_R_64F,
                                     A,
                                     lda,
                                     HIP_R_64F,
                                     W,
                                     HIP_R_64F,
                                     workspaceInBytesOnDevice,
                                     workspaceInBytesOnHost);
}

template <typename T>
hipsolverStatus_t cusolverDnxsyevd(  // NOLINT
  hipsolverHandle_t handle,
  cusolverDnParams_t params,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int64_t n,
  T* A,
  int64_t lda,
  T* W,
  T* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  T* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  hipStream_t stream);

template <>
inline hipsolverStatus_t cusolverDnxsyevd(  // NOLINT
  hipsolverHandle_t handle,
  cusolverDnParams_t params,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int64_t n,
  float* A,
  int64_t lda,
  float* W,
  float* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  float* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return cusolverDnXsyevd(handle,
                          params,
                          jobz,
                          uplo,
                          n,
                          HIP_R_32F,
                          A,
                          lda,
                          HIP_R_32F,
                          W,
                          HIP_R_32F,
                          bufferOnDevice,
                          workspaceInBytesOnDevice,
                          bufferOnHost,
                          workspaceInBytesOnHost,
                          info);
}

template <>
inline hipsolverStatus_t cusolverDnxsyevd(  // NOLINT
  hipsolverHandle_t handle,
  cusolverDnParams_t params,
  hipsolverEigMode_t jobz,
  hipblasFillMode_t uplo,
  int64_t n,
  double* A,
  int64_t lda,
  double* W,
  double* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  double* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  hipStream_t stream)
{
  RAFT_CUSOLVER_TRY(hipsolverSetStream(handle, stream));
  return cusolverDnXsyevd(handle,
                          params,
                          jobz,
                          uplo,
                          n,
                          HIP_R_64F,
                          A,
                          lda,
                          HIP_R_64F,
                          W,
                          HIP_R_64F,
                          bufferOnDevice,
                          workspaceInBytesOnDevice,
                          bufferOnHost,
                          workspaceInBytesOnHost,
                          info);
}
/** @} */
#endif

}  // namespace detail
}  // namespace linalg
}  // namespace raft
