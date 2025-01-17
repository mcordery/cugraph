#=============================================================================
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

if( (NOT DEFINED USE_CUDA) AND (NOT DEFINED USE_HIP) )
    message(FATAL_ERROR, "Build architecture is not set by default. You must set either -DUSE_CUDA or -DUSE_HIP")
endif()


if( USE_CUDA )
    project(CUGRAPH  LANGUAGES C CXX CUDA)
    include(${CMAKE_CURRENT_SOURCE_DIR}/build_CUDA.cmake)
endif()

if( USE_HIP )
    project(ROCGRAPH  LANGUAGES C CXX HIP)
    include(${CMAKE_CURRENT_SOURCE_DIR}/build_HIP.cmake)
endif()