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

#include <rmm/device_uvector.hpp>

#include <hipcub/hipcub.hpp>

namespace raft {

/**
 * @brief Convenience wrapper over cub's SortPairs method
 * @tparam KeyT key type
 * @tparam ValueT value type
 * @param workspace workspace buffer which will get resized if not enough space
 * @param inKeys input keys array
 * @param outKeys output keys array
 * @param inVals input values array
 * @param outVals output values array
 * @param len array length
 * @param stream cuda stream
 */
template <typename KeyT, typename ValueT>
void sortPairs(rmm::device_uvector<char>& workspace,
               const KeyT* inKeys,
               KeyT* outKeys,
               const ValueT* inVals,
               ValueT* outVals,
               int len,
               hipStream_t stream)
{
  size_t worksize = 0;  //  Fix 'worksize' may be used uninitialized in this function.
  hipcub::DeviceRadixSort::SortPairs(
    nullptr, worksize, inKeys, outKeys, inVals, outVals, len, 0, sizeof(KeyT) * 8, stream);
  workspace.resize(worksize, stream);
  hipcub::DeviceRadixSort::SortPairs(
    workspace.data(), worksize, inKeys, outKeys, inVals, outVals, len, 0, sizeof(KeyT) * 8, stream);
}

}  // namespace raft
