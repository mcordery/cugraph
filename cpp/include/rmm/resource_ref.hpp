/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <hip/memory_resource>

namespace rmm {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Alias for a `hip::mr::resource_ref` with the property
 * `hip::mr::device_accessible`.
 */
using device_resource_ref = hip::mr::resource_ref<hip::mr::device_accessible>;

/**
 * @brief Alias for a `hip::mr::async_resource_ref` with the property
 * `hip::mr::device_accessible`.
 */
using device_async_resource_ref = hip::mr::async_resource_ref<hip::mr::device_accessible>;

/**
 * @brief Alias for a `hip::mr::resource_ref` with the property
 * `hip::mr::host_accessible`.
 */
using host_resource_ref = hip::mr::resource_ref<hip::mr::host_accessible>;

/**
 * @brief Alias for a `hip::mr::async_resource_ref` with the property
 * `hip::mr::host_accessible`.
 */
using host_async_resource_ref = hip::mr::async_resource_ref<hip::mr::host_accessible>;

/**
 * @brief Alias for a `hip::mr::resource_ref` with the properties
 * `hip::mr::host_accessible` and `hip::mr::device_accessible`.
 */
using host_device_resource_ref =
  hip::mr::resource_ref<hip::mr::host_accessible, hip::mr::device_accessible>;

/**
 * @brief Alias for a `hip::mr::async_resource_ref` with the properties
 * `hip::mr::host_accessible` and `hip::mr::device_accessible`.
 */
using host_device_async_resource_ref =
  hip::mr::async_resource_ref<hip::mr::host_accessible, hip::mr::device_accessible>;

/** @} */  // end of group
}  // namespace rmm
