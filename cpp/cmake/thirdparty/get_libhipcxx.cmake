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


set(CUGRAPH_MIN_VERSION_libhipcxx "24.06.00")
set(CUGRAPH_BRANCH_VERSION_libhipcxx "24.06")
#set(CUGRAPH_MIN_VERSION_libhipcxx "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}.00")
#set(CUGRAPH_BRANCH_VERSION_libhipcxx "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}")


function(find_and_configure_libhipcxx)

    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN USE_LIBHIPCXX_STATIC COMPILE_LIBHIPCXX_LIB)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUGRAPH_BRANCH_VERSION_libhipcxx}")
        message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning libhipcxx locally.")
        set(CPM_DOWNLOAD_libhipcxx ON)
    elseif(PKG_USE_LIBHIPCXX_STATIC AND (NOT CPM_libhipcxx_SOURCE))
      message(STATUS "CUGRAPH: Cloning libhipcxx locally to build static libraries.")
      set(CPM_DOWNLOAD_libhipcxx ON)
    endif()

    if(PKG_COMPILE_LIBHIPCXX_LIB)
      if(NOT PKG_USE_LIBHIPCXX_STATIC)
        string(APPEND LIBHIPCXX_COMPONENTS " compiled")
      else()
        string(APPEND LIBHIPCXX_COMPONENTS " compiled_static")
      endif()
    endif()

    rapids_cpm_find(libhipcxx ${PKG_VERSION}
      GLOBAL_TARGETS      libhipcxx::libhipcxx
      BUILD_EXPORT_SET    cugraph-exports
      INSTALL_EXPORT_SET  cugraph-exports
      COMPONENTS ${LIBHIPCXX_COMPONENTS}
        CPM_ARGS
            EXCLUDE_FROM_ALL TRUE
            GIT_REPOSITORY
            GIT_REPOSITORY https://github.com/ROCm/libhipcxx
            GIT_TAG main
            # GIT_TAG f75154aa3be100dafb5c8b570f74198e51bb510a
            UPDATE_COMMAND git stash --all
            PATCH_COMMAND git apply
                          ${PROJECT_SOURCE_DIR}/deps/patch/libhipcxx_hip_tsc.patch)
                      SOURCE_SUBDIR  cpp
            OPTIONS
                "LIBHIPCXX_COMPILE_LIBRARY ${PKG_COMPILE_LIBHIPCXX_LIB}"
                "BUILD_TESTS OFF"
                "BUILD_BENCH OFF"
                "BUILD_CAGRA_HNSWLIB OFF"
    )

    if(libhipcxx_ADDED)
        message(VERBOSE "CUGRAPH: Using LIBHIPCXX located in ${libhipcxx_SOURCE_DIR}")
    else()
        message(VERBOSE "CUGRAPH: Using LIBHIPCXX located in ${libhipcxx_DIR}")
    endif()

endfunction()

# Change pinned tag and fork here to test a commit in CI
# To use a different LIBHIPCXX locally, set the CMake variable
# CPM_libhipcxx_SOURCE=/path/to/local/libhipcxx
find_and_configure_libhipcxx(VERSION    ${CUGRAPH_MIN_VERSION_libhipcxx}
                        FORK       rapidsai
                        PINNED_TAG branch-${CUGRAPH_BRANCH_VERSION_libhipcxx}

                        # When PINNED_TAG above doesn't match cugraph,
                        # force local libhipcxx clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        USE_LIBHIPCXX_STATIC ${USE_LIBHIPCXX_STATIC}
                        COMPILE_LIBHIPCXX_LIB ${CUGRAPH_COMPILE_LIBHIPCXX_LIB}
                        )
