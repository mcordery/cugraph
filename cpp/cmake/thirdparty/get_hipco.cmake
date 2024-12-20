#=============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

message(STATUS "Manually setting cugraph version since we're hacking things")


set(CUGRAPH_MIN_VERSION_hipco "24.06.00")
set(CUGRAPH_BRANCH_VERSION_hipco "24.06")
#set(CUGRAPH_MIN_VERSION_hipco "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}.00")
#set(CUGRAPH_BRANCH_VERSION_hipco "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}")


function(find_and_configure_hipco)

    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN USE_HIPCO_STATIC COMPILE_HIPCO_LIB)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUGRAPH_BRANCH_VERSION_hipco}")
        message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning hipco locally.")
        set(CPM_DOWNLOAD_hipco ON)
    elseif(PKG_USE_HIPCO_STATIC AND (NOT CPM_hipco_SOURCE))
      message(STATUS "CUGRAPH: Cloning hipco locally to build static libraries.")
      set(CPM_DOWNLOAD_hipco ON)
    endif()

    if(PKG_COMPILE_HIPCO_LIB)
      if(NOT PKG_USE_HIPCO_STATIC)
        string(APPEND HIPCO_COMPONENTS " compiled")
      else()
        string(APPEND HIPCO_COMPONENTS " compiled_static")
      endif()
    endif()

    rapids_cpm_find(hipco ${PKG_VERSION}
      GLOBAL_TARGETS      hipco::hipco
      BUILD_EXPORT_SET    cugraph-exports
      INSTALL_EXPORT_SET  cugraph-exports
      COMPONENTS ${HIPCO_COMPONENTS}
        CPM_ARGS
            EXCLUDE_FROM_ALL TRUE
            GIT_REPOSITORY
            GIT_REPOSITORY https://github.com/ROCm/hipCollections
            GIT_TAG branch-23.10-rocm-6.2
            # GIT_TAG 4a4b25eb4302a9657b7349ed524ed3ef08974ff8
            SOURCE_SUBDIR  cpp
            OPTIONS
                "HIPCO_COMPILE_LIBRARY ${PKG_COMPILE_HIPCO_LIB}"
                "BUILD_TESTS OFF"
                "BUILD_BENCH OFF"
                "BUILD_CAGRA_HNSWLIB OFF"
    )

    if(hipco_ADDED)
        message(VERBOSE "CUGRAPH: Using HIPCO located in ${hipco_SOURCE_DIR}")
    else()
        message(VERBOSE "CUGRAPH: Using HIPCO located in ${hipco_DIR}")
    endif()

endfunction()

# Change pinned tag and fork here to test a commit in CI
# To use a different HIPCO locally, set the CMake variable
# CPM_hipco_SOURCE=/path/to/local/hipco
find_and_configure_hipco(VERSION    ${CUGRAPH_MIN_VERSION_hipco}
                        FORK       rapidsai
                        PINNED_TAG branch-${CUGRAPH_BRANCH_VERSION_hipco}

                        # When PINNED_TAG above doesn't match cugraph,
                        # force local hipco clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        USE_HIPCO_STATIC ${USE_HIPCO_STATIC}
                        COMPILE_HIPCO_LIB ${CUGRAPH_COMPILE_HIPCO_LIB}
                        )
