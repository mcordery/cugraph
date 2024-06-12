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


message(STATUS "Configuring build for ${PROJECT_NAME}")

set(CMAKE_PREFIX_PATH /opt/rocm/lib/cmake)
enable_language(HIP)
include_directories(${HIP_INCLUDE_DIRS})

#if(NOT (DEFINED CMAKE_HIP_ARCHITECTURES))
#    set(CMAKE_HIP_ARCHITECTURES "gfx940;gfx941;gfx942")
#endif()

# Get ROCm CMake Helpers onto your CMake Module Path
if (NOT DEFINED ROCM_PATH )
if (NOT DEFINED ENV{ROCM_PATH} )
set(ROCM_PATH "/opt/rocm-6.1.2" CACHE PATH "ROCm path")
else()
set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm path")
endif()
endif()
set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake" ${CMAKE_MODULE_PATH})

#
# Find HIP
#


find_package(HIP REQUIRED CONFIG PATHS ${HIP_PATH} )
if (HIP_FOUND)
    message(STATUS "HIP_VERSION ${HIP_VERSION}")

    if( HIP_VERSION VERSION_LESS 6.0)  
        message(FATAL_ERROR "HIP compiler version must be at least 6.0")
    endif()   
    message(STATUS "HIP Toolkit found: ${HIP_VERSION}")
else()
    message(FATAL_ERROR "HIP Toolkit not found")
endif()

#
# Find hipcub
#

find_package(hipcub REQUIRED CONFIG PATHS ${ROCM_PATH}/hipcub )
if (hipcub_FOUND)
    message(STATUS "hipcub_VERSION ${hipcub_VERSION}")

    if( hipcub_VERSION VERSION_LESS 3.1)
        message(FATAL_ERROR "hipcub version must be at least 3.1")
    endif()
    message(STATUS "hipcub found: ${hipcub_VERSION}")
else()
    message(FATAL_ERROR "hipcub not found")
endif()

#
# Find rocprim
#

find_package(rocprim REQUIRED CONFIG PATHS ${ROCM_PATH}/rocprim )
if (rocprim_FOUND)
    message(STATUS "rocprim_VERSION ${rocprim_VERSION}")

    if( rocprim_VERSION VERSION_LESS 3.1)
        message(FATAL_ERROR "rocprim version must be at least 3.1")
    endif()
    message(STATUS "rocprim found: ${rocprim_VERSION}")
else()
    message(FATAL_ERROR "rocprim not found")
endif()



#
# Find rocthrust
#

find_package(rocthrust REQUIRED CONFIG PATHS ${ROCM_PATH}/rocthrust )
if (rocthrust_FOUND)
    message(STATUS "rocthrust_VERSION ${rocthrust_VERSION}")

    if( rocthrust_VERSION VERSION_LESS 3.0)
        message(FATAL_ERROR "rocthrust version must be at least 3.0")
    endif()
    message(STATUS "rocthrust found: ${rocthrust_VERSION}")
else()
    message(FATAL_ERROR "rocthrust not found")
endif()



#
# Find ROCM hipblas
#

find_package(hipblas REQUIRED CONFIG PATHS ${ROCM_PATH}/hipblas )
if (hipblas_FOUND)
    message(STATUS "hipblas_VERSION ${hipblas_VERSION}")

    if( hipblas_VERSION VERSION_LESS 2.1)
        message(FATAL_ERROR "hipblas version must be at least 2.1")
    endif()
    message(STATUS "hipblas found: ${hipblas_VERSION}")
else()
    message(FATAL_ERROR "hipblas not found")
endif()

#
# Find ROCM hipsparse
#

find_package(hipsparse REQUIRED CONFIG PATHS ${ROCM_PATH}/hipsparse  ) 
if (hipsparse_FOUND)
    message(STATUS "hipsparse_VERSION ${hipsparse_VERSION}")

    if( hipsparse_VERSION VERSION_LESS 3.0)
        message(FATAL_ERROR "hipsparse version must be at least 3.0")
    endif()
    message(STATUS "hipsparse found: ${hipsparse_VERSION}")
else()
    message(FATAL_ERROR "hipsparse not found")
endif()

#
# Find ROCM hiprand
#

find_package(hiprand REQUIRED CONFIG PATHS ${ROCM_PATH}/hiprand )
if (hiprand_FOUND)
    message(STATUS "hiprand_VERSION ${hiprand_VERSION}")

    if( hiprand_VERSION VERSION_LESS 2.10)
        message(FATAL_ERROR "hiprand version must be at least 2.10")
    endif()
    message(STATUS "hiprand found: ${hiprand_VERSION}")
else()
    message(FATAL_ERROR "hiprand not found")
endif()

#
# Find ROCM hipsolver
#

find_package(hipsolver REQUIRED CONFIG PATHS ${ROCM_PATH}/hipsolver )
if (hipsolver_FOUND)
    message(STATUS "hipsolver_VERSION ${hipsolver_VERSION}")

    if( hipsolver_VERSION VERSION_LESS 2.1)
        message(FATAL_ERROR "hipsolver version must be at least 2.1")
    endif()
    message(STATUS "hipsolver found: ${hipsolver_VERSION}")
else()
    message(FATAL_ERROR "hipsolver not found")
endif()

#
# Find libhipcxx
#

find_package(libhipcxx REQUIRED CONFIG PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include/libhipcxx/install/lib/cmake/libhipcxx )
if (libhipcxx_FOUND)
    message(STATUS "libhipcxx_VERSION ${libhipcxx_VERSION}")

    if( libhipcxx_VERSION VERSION_LESS 1.9)
        message(FATAL_ERROR "libhipcxx version must be at least 1.9")
    endif()
    message(STATUS "libhipcxx found: ${libhipcxx_VERSION}")
else()
    message(FATAL_ERROR "libhipcxx not found")
endif()


if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
    message(FATAL_ERROR "GCC compiler must be at least 17.0")
endif()
#set(CMAKE_HIP_ARCHITECTURES "gfx900")

######### Set build configuration ############

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build configuration")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build configuration")
else()
    message(STATUS "No build configuration set, assuming release configuration")
    set(CMAKE_BUILD_TYPE "Release")
endif() 


################################################################################
# - User Options  --------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build rocGraph shared libraries" ON)
option(BUILD_ROCGRAPH_MG_TESTS "Build rocGraph multigpu algorithm tests" OFF)
option(CMAKE_HIP_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(BUILD_TESTS "Configure CMake to build tests" OFF)
option(ROCGRAPH_COMPILE_RAFT_LIB "Compile the raft library instead of using it header-only" OFF)
option(HIP_STATIC_RUNTIME "Statically link the HIP toolkit runtime and libraries" OFF)

message(VERBOSE "ROCGRAPH: HIP_STATIC_RUNTIME=${HIP_STATIC_RUNTIME}")

################################################################################
# - compiler options -----------------------------------------------------------

#
# NB check the flags here
#

set(ROCGRAPH_CXX_FLAGS -DNO_CUGRAPH_OPS -DCUTLASS_NAMESPACE=raft_cutlass -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DSPDLOG_FMT_EXTERNAL -DTHRUST_DISABLE_ABI_NAMESPACE  -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -Drocgraph_EXPORTS)
set(ROCGRAPH_HIP_FLAGS "")

set(THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP) 
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_HIP) 

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND ROCGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif()

#message("-- Building for GPU_ARCHS = ${CMAKE_HIP_ARCHITECTURES}")

list(APPEND ROCGRAPH_CXX_FLAGS --offload-arch=gfx942) # need to specify this because I'm building on a machine without an amd card and libhipcxx needs to know for the timers
list(APPEND ROCGRAPH_CXX_FLAGS  -DFMT_HEADER_ONLY -DUSE_LIBHIPCXX_PRT )

#list(APPEND ROCGRAPH_CXX_FLAGS "-stdlib=libc++") # need to specify this because I'm building on a machine without an amd card and libhipcxx needs to know for the timers

message(STATUS "HID ${HIP_INCLUDE_DIRS}")

list(APPEND ROCGRAPH_CXX_FLAGS -Wno-unused-result)

list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS})

list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/libhipcxx/install/include ) # for hipco

list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS}/rocprim)
list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS}/hipcub)

list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS}/hipblas)
list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS}/hipsparse)
list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS}/hiprand)
list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS}/hipsolver)
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/rmm/ )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/spdlog/include )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/fmt/include )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/raft/include )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass/include )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/hipco/include )
list(APPEND ROCGRAPH_CXX_FLAGS  -cxx-isystem ${CMAKE_CURRENT_SOURCE_DIR}/include )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/src/ )



# Option to enable line info in HIP device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_HIP_LINEINFO)
    list(APPEND ROCGRAPH_HIP_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND ROCGRAPH_HIP_FLAGS -G -Xcompiler=-rdynamic)
endif()


###################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------

if (BUILD_ROCGRAPH_MTMG_TESTS)
 # include(cmake/thirdparty/get_ucp.cmake)
endif()

if(BUILD_TESTS)
#  include(${rapids-cmake-dir}/cpm/gtest.cmake)
#  rapids_cpm_gtest(BUILD_STATIC)
endif()

################################################################################
# - librocgraph library target --------------------------------------------------

# NOTE: The most expensive compilations are listed first
#       since ninja will run them in parallel in this order,
#       which should give us a better parallel schedule.

set(ROCGRAPH_SOURCES
      src/utilities/shuffle_vertices.cpp
      src/detail/permute_range.cpp
      src/utilities/shuffle_vertex_pairs.cpp
      src/detail/collect_local_vertex_values.cpp
      src/detail/groupby_and_count.cpp
      src/detail/collect_comm_wrapper.cpp
      src/sampling/random_walks_mg.cpp      
      #src/community/detail/common_methods_mg.cpp
      #src/community/detail/common_methods_sg.cpp
      #src/community/detail/refine_sg.cpp
      #src/community/detail/refine_mg.cpp
      src/community/edge_triangle_count_sg.cpp
      src/community/detail/maximal_independent_moves_sg.cpp
      src/community/detail/maximal_independent_moves_mg.cpp
      src/detail/utility_wrappers.cpp
      #src/structure/graph_view_mg.cpp
      src/structure/remove_self_loops.cpp
      src/structure/remove_multi_edges.cpp
      src/utilities/path_retrieval.cpp
      src/structure/legacy/graph.cpp
      src/linear_assignment/legacy/hungarian.cpp
      #src/link_prediction/jaccard_sg.cpp
      #src/link_prediction/sorensen_sg.cpp
      #src/link_prediction/overlap_sg.cpp
      #src/link_prediction/jaccard_mg.cpp
      #src/link_prediction/sorensen_mg.cpp
      #src/link_prediction/overlap_mg.cpp
      #src/layout/legacy/force_atlas2.cpp
      src/converters/legacy/COOtoCSR.cpp
      #src/community/legacy/spectral_clustering.cpp
      src/community/louvain_sg.cpp
      src/community/louvain_mg.cpp
      src/community/leiden_sg.cpp
      src/community/leiden_mg.cpp
      src/community/ecg_sg.cpp
      src/community/ecg_mg.cpp
      src/community/legacy/louvain.cpp
      src/community/legacy/ecg.cpp
      src/community/egonet_sg.cpp
      src/community/egonet_mg.cpp
      #src/community/k_truss_sg.cpp
      #src/sampling/random_walks.cpp
      src/sampling/random_walks_sg.cpp
      src/sampling/detail/prepare_next_frontier_sg.cpp
      src/sampling/detail/prepare_next_frontier_mg.cpp
      src/sampling/detail/gather_one_hop_edgelist_sg.cpp
      src/sampling/detail/gather_one_hop_edgelist_mg.cpp
      src/sampling/detail/remove_visited_vertices_from_frontier.cpp
      src/sampling/detail/sample_edges_sg.cpp
      src/sampling/detail/sample_edges_mg.cpp
      src/sampling/detail/shuffle_and_organize_output_mg.cpp
      src/sampling/uniform_neighbor_sampling_mg.cpp
      src/sampling/uniform_neighbor_sampling_sg.cpp
      #src/sampling/renumber_sampled_edgelist_sg.cpp
      #src/sampling/sampling_post_processing_sg.cpp
      #src/cores/core_number_sg.cpp
      #src/cores/core_number_mg.cpp
      src/cores/k_core_sg.cpp
      src/cores/k_core_mg.cpp
      src/components/legacy/connectivity.cpp
      src/generators/generate_rmat_edgelist.cpp
      src/generators/generate_bipartite_rmat_edgelist.cpp
      #src/generators/generator_tools.cpp
      src/generators/simple_generators.cpp
      src/generators/erdos_renyi_generator.cpp
      src/structure/graph_sg.cpp
      src/structure/graph_mg.cpp
      #src/structure/graph_view_sg.cpp
      src/structure/decompress_to_edgelist_sg.cpp
      src/structure/decompress_to_edgelist_mg.cpp
      src/structure/symmetrize_graph_sg.cpp
      src/structure/symmetrize_graph_mg.cpp
      src/structure/transpose_graph_sg.cpp
      src/structure/transpose_graph_mg.cpp
      src/structure/transpose_graph_storage_sg.cpp
      src/structure/transpose_graph_storage_mg.cpp
      src/structure/coarsen_graph_sg.cpp
      src/structure/coarsen_graph_mg.cpp
      src/structure/graph_weight_utils_mg.cpp
      src/structure/graph_weight_utils_sg.cpp
      #src/structure/renumber_edgelist_sg.cpp
      #src/structure/renumber_edgelist_mg.cpp
      #src/structure/renumber_utils_sg.cpp
      #src/structure/renumber_utils_mg.cpp
      #src/structure/relabel_sg.cpp
      #src/structure/relabel_mg.cpp
      #src/structure/induced_subgraph_sg.cpp
      #src/structure/induced_subgraph_mg.cpp
      src/structure/select_random_vertices_sg.cpp
      src/structure/select_random_vertices_mg.cpp
      src/traversal/extract_bfs_paths_sg.cpp
      src/traversal/extract_bfs_paths_mg.cpp
      #src/traversal/bfs_sg.cpp
      #src/traversal/bfs_mg.cpp
      #src/traversal/sssp_sg.cpp
      #src/traversal/od_shortest_distances_sg.cpp
      #src/traversal/sssp_mg.cpp
      src/link_analysis/hits_sg.cpp
      src/link_analysis/hits_mg.cpp
      #src/link_analysis/pagerank_sg.cpp
      #src/link_analysis/pagerank_mg.cpp
      src/centrality/katz_centrality_sg.cpp
      src/centrality/katz_centrality_mg.cpp
      #src/centrality/eigenvector_centrality_sg.cpp
      #src/centrality/eigenvector_centrality_mg.cpp
      #src/centrality/betweenness_centrality_sg.cpp
      #src/centrality/betweenness_centrality_mg.cpp
      src/tree/legacy/mst.cpp
      #src/components/weakly_connected_components_sg.cpp
      #src/components/weakly_connected_components_mg.cpp
      src/components/mis_sg.cpp
      src/components/mis_mg.cpp
      src/components/vertex_coloring_sg.cpp
      src/components/vertex_coloring_mg.cpp
      src/structure/create_graph_from_edgelist_sg.cpp
      src/structure/create_graph_from_edgelist_mg.cpp
      src/structure/symmetrize_edgelist_sg.cpp
      src/structure/symmetrize_edgelist_mg.cpp
      #src/community/triangle_count_sg.cpp
      #src/community/triangle_count_mg.cpp
      #src/traversal/k_hop_nbrs_sg.cpp
      #src/traversal/k_hop_nbrs_mg.cpp
     src/mtmg/vertex_result.cpp
)


add_library(rocgraph ${ROCGRAPH_SOURCES})

    # With ROCM 6.1.2 currently, do NO change C++ std to anything higher than 17 if you have #include <chrono> in a source/header file

    set_target_properties(rocgraph
        PROPERTIES BUILD_RPATH                         "\$ORIGIN"
                INSTALL_RPATH                       "\$ORIGIN"
                # set target compile options
                CXX_STANDARD                       17
                CXX_STANDARD_REQUIRED               ON
                CXX_EXTENSIONS                     ON
                HIP_STANDARD                       17
                HIP_STANDARD_REQUIRED              ON
                HIP_EXTENSIONS                     ON
                POSITION_INDEPENDENT_CODE           ON
                INTERFACE_POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(rocgraph 
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ROCGRAPH_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:HIP>:${ROCGRAPH_HIP_FLAGS}>"
    )

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(rocgraph PUBLIC HIP_API_PER_THREAD_DEFAULT_STREAM)

##file(WRITE "${ROCGRAPH_BINARY_DIR}/fatbin.ld"
##[=[
##SECTIONS
##{
##  .nvFatBinSegment : { *(.nvFatBinSegment) }
##  .nv_fatbin : { *(.nv_fatbin) }
##}
##]=])
##target_link_options(rocgraph PRIVATE "${ROCGRAPH_BINARY_DIR}/fatbin.ld")

add_library(rocgraph::rocgraph ALIAS rocgraph)

################################################################################
# - include paths --------------------------------------------------------------

target_include_directories(rocgraph
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

################################################################################

target_link_libraries(rocgraph
    PUBLIC
        hip::host
        hip::device
        hipblas
        hipsparse
        hiprand
        hipsolver
 #       libhipcxx::libhipcxx
    )

################################################################################
# - C-API library --------------------------------------------------------------

add_library(rocgraph_c
        src/c_api/resource_handle.cpp
        src/c_api/array.cpp
        src/c_api/degrees.cpp
        src/c_api/degrees_result.cpp
        src/c_api/error.cpp
        src/c_api/graph_sg.cpp
        src/c_api/graph_mg.cpp
        src/c_api/graph_functions.cpp
        src/c_api/pagerank.cpp
        src/c_api/katz.cpp
        src/c_api/centrality_result.cpp
        src/c_api/eigenvector_centrality.cpp
        src/c_api/betweenness_centrality.cpp
        src/c_api/core_number.cpp
        src/c_api/k_truss.cpp
        src/c_api/core_result.cpp
        src/c_api/extract_ego.cpp
        src/c_api/ecg.cpp
        src/c_api/k_core.cpp
        src/c_api/hierarchical_clustering_result.cpp
        src/c_api/induced_subgraph.cpp
        src/c_api/capi_helper.cpp
        src/c_api/legacy_spectral.cpp
        src/c_api/legacy_ecg.cpp
        src/c_api/graph_helper_sg.cpp
        src/c_api/graph_helper_mg.cpp
        src/c_api/graph_generators.cpp
        src/c_api/induced_subgraph_result.cpp
        src/c_api/hits.cpp

        src/c_api/bfs.cpp
        src/c_api/sssp.cpp
        src/c_api/extract_paths.cpp
        src/c_api/random_walks.cpp
        src/c_api/random.cpp
        src/c_api/similarity.cpp
        src/c_api/leiden.cpp
        src/c_api/louvain.cpp
        src/c_api/triangle_count.cpp
        src/c_api/uniform_neighbor_sampling.cpp
        src/c_api/labeling_result.cpp
        src/c_api/weakly_connected_components.cpp
        src/c_api/strongly_connected_components.cpp
        src/c_api/allgather.cpp
        )
add_library(rocgraph::rocgraph_c ALIAS rocgraph_c)

# Currently presuming we aren't calling any HIP kernels in rocgraph_c

set_target_properties(rocgraph_c
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        20
               CXX_STANDARD_REQUIRED               ON
               HIP_STANDARD                       20
               HIP_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(rocgraph_c
             PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ROCGRAPH_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:HIP>:${ROCGRAPH_HIP_FLAGS}>"
)

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(rocgraph_c PUBLIC HIP_API_PER_THREAD_DEFAULT_STREAM)

#target_link_options(rocgraph_c PRIVATE "${ROCGRAPH_BINARY_DIR}/fatbin.ld")

################################################################################
# - C-API include paths --------------------------------------------------------

target_include_directories(rocgraph_c
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)


################################################################################
# - C-API link libraries -------------------------------------------------------
target_link_libraries(rocgraph_c PRIVATE rocgraph::rocgraph)

################################################################################
# - generate tests -------------------------------------------------------------

#if(BUILD_TESTS)
#  include(CTest)
#  add_subdirectory(tests)
#endif()

################################################################################
# - install targets ------------------------------------------------------------
set(libdir $${CMAKE_CURRENT_BINARY_DIR})
include(CPack)

install(TARGETS rocgraph
        DESTINATION ${lib_dir}
        )

install(DIRECTORY include/rocgraph/
        DESTINATION include/rocgraph)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/rocgraph/version_config.hpp
        DESTINATION include/rocgraph)

install(TARGETS rocgraph_c
        DESTINATION ${lib_dir}
        )

install(DIRECTORY include/rocgraph_c/
        DESTINATION include/rocgraph_c)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/rocgraph_c/version_config.hpp
        DESTINATION include/rocgraph_c)

################################################################################
# - install export -------------------------------------------------------------

set(doc_string
[=[
Provide targets for rocGraph.

rocGraph library is a collection of GPU accelerated graph algorithms that process data found in
[GPU DataFrames](https://github.com/rapidsai/cudf).

]=])

#rapids_export(INSTALL rocgraph
#    EXPORT_SET rocgraph-exports
#    GLOBAL_TARGETS rocgraph rocgraph_c
#    NAMESPACE rocgraph::
#    DOCUMENTATION doc_string
#    )

################################################################################
# - build export ---------------------------------------------------------------
#rapids_export(BUILD rocgraph
#    EXPORT_SET rocgraph-exports
#    GLOBAL_TARGETS rocgraph rocgraph_c
#    NAMESPACE rocgraph::
#    DOCUMENTATION doc_string
#    )

################################################################################
# - make documentation ---------------------------------------------------------
# requires doxygen and graphviz to be installed
# from build directory, run make docs_rocgraph

# doc targets for rocgraph
find_package(Doxygen 1.8.11)
if(Doxygen_FOUND)
    add_custom_command(OUTPUT ROCGRAPH_DOXYGEN
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                       COMMAND ${CMAKE_COMMAND} -E env "RAPIDS_VERSION_MAJOR_MINOR=${RAPIDS_VERSION_MAJOR_MINOR}" doxygen Doxyfile
                       VERBATIM)

    add_custom_target(docs_rocgraph DEPENDS ROCGRAPH_DOXYGEN)
endif()
