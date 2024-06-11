//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_block_tag) {
    __threadfence_block();
}

template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_block_tag) {
    *__ret = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_block_tag) {
    *__ret = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_block_tag) {
    __hip_atomic_store(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_block_tag) {
    __hip_atomic_store(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_block_tag) {
    return __hip_atomic_compare_exchange_weak(__ptr, __expected, *__desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_block_tag) {
    *__ret = __hip_atomic_exchange(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_block_tag) {
    return __hip_atomic_compare_exchange_weak(__ptr, __expected, *__desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_block_tag) {
    *__ret = __hip_atomic_exchange(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);    
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_device_tag) {
    __threadfence();
}

template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_device_tag) {
    *__ret = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_device_tag) {
    *__ret = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_device_tag) {
    __hip_atomic_store(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_device_tag) {
    __hip_atomic_store(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);    
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_device_tag) {
    return __hip_atomic_compare_exchange_weak(__ptr, __expected, *__desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_device_tag) {
    *__ret = __hip_atomic_exchange(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_device_tag) {
    return __hip_atomic_compare_exchange_weak(__ptr, __expected, *__desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_device_tag) {
    *__ret = __hip_atomic_exchange(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);    
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}
template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_system_tag) {
    __threadfence_system();
}

template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_system_tag) {
    *__ret = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_system_tag) {
    *__ret = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_system_tag) {
    __hip_atomic_store(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_system_tag) {
    __hip_atomic_store(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool __is_weak, int __success_memorder, int __failure_memorder, __thread_scope_system_tag) {
    if(__is_weak)
        return __hip_atomic_compare_exchange_weak(__ptr, __expected, *__desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_SYSTEM);
    else 
        return __hip_atomic_compare_exchange_strong(__ptr, __expected, *__desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_system_tag) {
    *__ret = __hip_atomic_exchange(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
   return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
   return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==4 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_system_tag) {
    return __hip_atomic_compare_exchange_weak(__ptr, __expected, *__desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_system_tag) {
    *__ret = __hip_atomic_exchange(__ptr, *__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_signed<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value && hip::std::is_unsigned<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_floating_point<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type, typename hip::std::enable_if<sizeof(_Type)==8 && hip::std::is_integral<_Type>::value, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}
template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}