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


set(CUGRAPH_MIN_VERSION_rmm "24.06.00")
set(CUGRAPH_BRANCH_VERSION_rmm "24.06")
#set(CUGRAPH_MIN_VERSION_rmm "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}.00")
#set(CUGRAPH_BRANCH_VERSION_rmm "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}")


function(find_and_configure_rmm)

    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN USE_RMM_STATIC COMPILE_RMM_LIB)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUGRAPH_BRANCH_VERSION_rmm}")
        message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning rmm locally.")
        set(CPM_DOWNLOAD_rmm ON)
    elseif(PKG_USE_RMM_STATIC AND (NOT CPM_rmm_SOURCE))
      message(STATUS "CUGRAPH: Cloning rmm locally to build static libraries.")
      set(CPM_DOWNLOAD_rmm ON)
    endif()

    if(PKG_COMPILE_RMM_LIB)
      if(NOT PKG_USE_RMM_STATIC)
        string(APPEND RMM_COMPONENTS " compiled")
      else()
        string(APPEND RMM_COMPONENTS " compiled_static")
      endif()
    endif()

    rapids_cpm_find(rmm ${PKG_VERSION}
      GLOBAL_TARGETS      rmm::rmm
      BUILD_EXPORT_SET    cugraph-exports
      INSTALL_EXPORT_SET  cugraph-exports
      COMPONENTS ${RMM_COMPONENTS}
        CPM_ARGS
            EXCLUDE_FROM_ALL TRUE
            GIT_REPOSITORY
            https://$ENV{GITHUB_USER}:$ENV{GITHUB_PASS}@github.com/AMD-AI/rmm-rocm
          # GIT_TAG e1e118700a60637aa62fc603634775d3ce3c87ef
            GIT_TAG feat/fixes_for_wip-24.06
            SOURCE_SUBDIR  cpp
            OPTIONS
                "RMM_COMPILE_LIBRARY ${PKG_COMPILE_RMM_LIB}"
                "BUILD_TESTS OFF"
                "BUILD_BENCH OFF"
                "BUILD_CAGRA_HNSWLIB OFF"
    )

    if(rmm_ADDED)
        message(VERBOSE "CUGRAPH: Using RMM located in ${rmm_SOURCE_DIR}")
    else()
        message(VERBOSE "CUGRAPH: Using RMM located in ${rmm_DIR}")
    endif()

endfunction()

# Change pinned tag and fork here to test a commit in CI
# To use a different RMM locally, set the CMake variable
# CPM_rmm_SOURCE=/path/to/local/rmm
find_and_configure_rmm(VERSION    ${CUGRAPH_MIN_VERSION_rmm}
                        FORK       rapidsai
                        PINNED_TAG branch-${CUGRAPH_BRANCH_VERSION_rmm}

                        # When PINNED_TAG above doesn't match cugraph,
                        # force local rmm clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        USE_RMM_STATIC ${USE_RMM_STATIC}
                        COMPILE_RMM_LIB ${CUGRAPH_COMPILE_RMM_LIB}
                        )
