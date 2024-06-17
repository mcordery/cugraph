//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_BOUNDED_ARRAY_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_BOUNDED_ARRAY_H

#ifndef __cuda_std__
#include <__config>
#include <cstddef>
#endif  // __cuda_std__

#include "../__type_traits/integral_constant.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class>
struct _LIBCUDACXX_TEMPLATE_VIS __libcpp_is_bounded_array : false_type {};
template <class _Tp, size_t _Np>
struct _LIBCUDACXX_TEMPLATE_VIS __libcpp_is_bounded_array<_Tp[_Np]> : true_type {};

#if _LIBCUDACXX_STD_VER > 11

template <class>
struct _LIBCUDACXX_TEMPLATE_VIS is_bounded_array : false_type {};
template <class _Tp, size_t _Np>
struct _LIBCUDACXX_TEMPLATE_VIS is_bounded_array<_Tp[_Np]> : true_type {};

template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_bounded_array_v = is_bounded_array<_Tp>::value;

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif  // _LIBCUDACXX___TYPE_TRAITS_IS_BOUNDED_ARRAY_H
