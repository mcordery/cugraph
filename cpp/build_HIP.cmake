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

#
# Find HIP
#

find_package(hip REQUIRED)
if (hip_FOUND)
    message(STATUS "hip_VERSION ${hip_VERSION}")

    if( hip_VERSION VERSION_LESS 6.0)  
        message(FATAL_ERROR "HIP compiler version must be at least 6.0")
    endif()   
    message(STATUS "HIP Toolkit found: ${hip_VERSION}")
else()
    message(FATAL_ERROR "HIP Toolkit not found")
endif()

#
# Find ROCM rocblas
#

find_package(rocblas REQUIRED)
if (rocblas_FOUND)
    message(STATUS "rocblas_VERSION ${rocblas_VERSION}")

    if( rocblas_VERSION VERSION_LESS 4.1)
        message(FATAL_ERROR "rocblas version must be at least 4.1")
    endif()
    message(STATUS "rocblas found: ${rocblas_VERSION}")
else()
    message(FATAL_ERROR "rocblas not found")
endif()

#
# Find ROCM rocsparse
#

find_package(rocsparse REQUIRED)
if (rocsparse_FOUND)
    message(STATUS "rocsparse_VERSION ${rocsparse_VERSION}")

    if( rocsparse_VERSION VERSION_LESS 3.1)
        message(FATAL_ERROR "rocsparse version must be at least 3.1")
    endif()
    message(STATUS "rocsparse found: ${rocsparse_VERSION}")
else()
    message(FATAL_ERROR "rocsparse not found")
endif()

#
# Find ROCM rocrand
#

find_package(rocrand REQUIRED)
if (rocrand_FOUND)
    message(STATUS "rocrand_VERSION ${rocrand_VERSION}")

    if( rocrand_VERSION VERSION_LESS 3.0)
        message(FATAL_ERROR "rocrand version must be at least 3.0")
    endif()
    message(STATUS "rocrand found: ${rocrand_VERSION}")
else()
    message(FATAL_ERROR "rocrand not found")
endif()

#
# Find ROCM rocsolver
#

find_package(rocsolver REQUIRED)
if (rocsolver_FOUND)
    message(STATUS "rocsolver_VERSION ${rocsolver_VERSION}")

    if( rocsolver_VERSION VERSION_LESS 3.25)
        message(FATAL_ERROR "rocsolver version must be at least 3.25")
    endif()
    message(STATUS "rocsolver found: ${rocsolver_VERSION}")
else()
    message(FATAL_ERROR "rocsolver not found")
endif()



if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
    message(FATAL_ERROR "GCC compiler must be at least 17.0")
endif()
set(CMAKE_HIP_ARCHITECTURES "gfx900")

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

set(ROCGRAPH_C_FLAGS "")
set(ROCGRAPH_CXX_FLAGS "")

#
# NB check the flags here
#
set(ROCGRAPH_CXX_FLAGS -DCUTLASS_NAMESPACE=raft_cutlass -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DSPDLOG_FMT_EXTERNAL -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_DISABLE_ABI_NAMESPACE -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -Drocgraph_EXPORTS)
set(ROCGRAPH_HIP_FLAGS -DCUTLASS_NAMESPACE=raft_cutlass -DLIBHIPCXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DSPDLOG_FMT_EXTERNAL -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP -DTHRUST_DISABLE_ABI_NAMESPACE -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -Drocgraph_EXPORTS)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND ROCGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_HIP_ARCHITECTURES}")


list(APPEND ROCGRAPH_C_FLAGS -DFMT_HEADER_ONLY )
list(APPEND ROCGRAPH_CXX_FLAGS  -DFMT_HEADER_ONLY )
#list(APPEND ROCGRAPH_HIP_FLAGS -DFMT_HEADER_ONLY -DTHRUST_IGNORE_CUB_VERSION_CHECK )
list(APPEND ROCGRAPH_HIP_FLAGS -DFMT_HEADER_ONLY  )



list(APPEND ROCGRAPH_HIP_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND ROCGRAPH_HIP_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
list(APPEND ROCGRAPH_HIP_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND ROCGRAPH_HIP_FLAGS -Xfatbin=-compress-all)

#list(APPEND ROCGRAPH_HIP_FLAGS -I/usr/local/cuda/include )
list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cccl/thrust )
list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cccl/cub/ )
list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cccl/cub/cub/ )
list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cccl/libcudacxx/include )
list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/rmm/ )
 list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/fmt/include )
 list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/spdlog/include )
 list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/raft/include )
 list(APPEND ROCGRAPH_HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass/include )

list(APPEND ROCGRAPH_CXX_FLAGS -DCUTLASS_NAMESPACE=raft_cutlass -DLIBHIPCXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DSPDLOG_FMT_EXTERNAL -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP -DTHRUST_DISABLE_ABI_NAMESPACE -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -Drocgraph_EXPORTS)
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cccl/thrust )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cccl/cub/ )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cccl/cub/cub/ )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cccl/libcudacxx/include )
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/rmm/ )
 list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/fmt/include )
 list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/spdlog/include )
 list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/raft/include )
 list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass/include )
 list(APPEND ROCGRAPH_CXX_FLAGS -I/usr/local/cuda/include )



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
      src/community/detail/common_methods_mg.cpp
      src/community/detail/common_methods_sg.cpp
      src/community/detail/refine_sg.cpp
      src/community/detail/refine_mg.cpp
      src/community/edge_triangle_count_sg.cpp
      src/community/detail/maximal_independent_moves_sg.cpp
      src/community/detail/maximal_independent_moves_mg.cpp
      src/detail/utility_wrappers.cpp
      src/structure/graph_view_mg.cpp
      src/structure/remove_self_loops.cpp
      src/structure/remove_multi_edges.cpp
      src/utilities/path_retrieval.cpp
      src/structure/legacy/graph.cpp
      src/linear_assignment/legacy/hungarian.cpp
      src/link_prediction/jaccard_sg.cpp
      src/link_prediction/sorensen_sg.cpp
      src/link_prediction/overlap_sg.cpp
      src/link_prediction/jaccard_mg.cpp
      src/link_prediction/sorensen_mg.cpp
      src/link_prediction/overlap_mg.cpp
      src/layout/legacy/force_atlas2.cpp
      src/converters/legacy/COOtoCSR.cpp
      src/community/legacy/spectral_clustering.cpp
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
      src/community/k_truss_sg.cpp
      src/sampling/random_walks.cpp
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
      src/sampling/renumber_sampled_edgelist_sg.cpp
      src/sampling/sampling_post_processing_sg.cpp
      src/cores/core_number_sg.cpp
      src/cores/core_number_mg.cpp
      src/cores/k_core_sg.cpp
      src/cores/k_core_mg.cpp
      src/components/legacy/connectivity.cpp
      src/generators/generate_rmat_edgelist.cpp
      src/generators/generate_bipartite_rmat_edgelist.cpp
      src/generators/generator_tools.cpp
      src/generators/simple_generators.cpp
      src/generators/erdos_renyi_generator.cpp
      src/structure/graph_sg.cpp
      src/structure/graph_mg.cpp
      src/structure/graph_view_sg.cpp
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
      src/structure/renumber_edgelist_sg.cpp
      src/structure/renumber_edgelist_mg.cpp
      src/structure/renumber_utils_sg.cpp
      src/structure/renumber_utils_mg.cpp
      src/structure/relabel_sg.cpp
      src/structure/relabel_mg.cpp
      src/structure/induced_subgraph_sg.cpp
      src/structure/induced_subgraph_mg.cpp
      src/structure/select_random_vertices_sg.cpp
      src/structure/select_random_vertices_mg.cpp
      src/traversal/extract_bfs_paths_sg.cpp
      src/traversal/extract_bfs_paths_mg.cpp
      src/traversal/bfs_sg.cpp
      src/traversal/bfs_mg.cpp
      src/traversal/sssp_sg.cpp
      src/traversal/od_shortest_distances_sg.cpp
      src/traversal/sssp_mg.cpp
      src/link_analysis/hits_sg.cpp
      src/link_analysis/hits_mg.cpp
      src/link_analysis/pagerank_sg.cpp
      src/link_analysis/pagerank_mg.cpp
      src/centrality/katz_centrality_sg.cpp
      src/centrality/katz_centrality_mg.cpp
      src/centrality/eigenvector_centrality_sg.cpp
      src/centrality/eigenvector_centrality_mg.cpp
      src/centrality/betweenness_centrality_sg.cpp
      src/centrality/betweenness_centrality_mg.cpp
      src/tree/legacy/mst.cpp
      src/components/weakly_connected_components_sg.cpp
      src/components/weakly_connected_components_mg.cpp
      src/components/mis_sg.cpp
      src/components/mis_mg.cpp
      src/components/vertex_coloring_sg.cpp
      src/components/vertex_coloring_mg.cpp
      src/structure/create_graph_from_edgelist_sg.cpp
      src/structure/create_graph_from_edgelist_mg.cpp
      src/structure/symmetrize_edgelist_sg.cpp
      src/structure/symmetrize_edgelist_mg.cpp
      src/community/triangle_count_sg.cpp
      src/community/triangle_count_mg.cpp
      src/traversal/k_hop_nbrs_sg.cpp
      src/traversal/k_hop_nbrs_mg.cpp
      src/mtmg/vertex_result.cpp
)


add_library(rocgraph ${ROCGRAPH_SOURCES})


    set_target_properties(rocgraph
        PROPERTIES BUILD_RPATH                         "\$ORIGIN"
                INSTALL_RPATH                       "\$ORIGIN"
                # set target compile options
                CXX_STANDARD                        17
                CXX_STANDARD_REQUIRED               ON
                ROCM_STANDARD                       17
                ROCM_STANDARD_REQUIRED              ON
                POSITION_INDEPENDENT_CODE           ON
                INTERFACE_POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(rocgraph
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ROCGRAPH_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:ROCM>:${ROCGRAPH_ROCM_FLAGS}>"
    )

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(rocgraph PUBLIC HIP_API_PER_THREAD_DEFAULT_STREAM)

#file(WRITE "${ROCGRAPH_BINARY_DIR}/fatbin.ld"
#[=[
#SECTIONS
#{
#  .nvFatBinSegment : { *(.nvFatBinSegment) }
#  .nv_fatbin : { *(.nv_fatbin) }
#}
#]=])
#target_link_options(rocgraph PRIVATE "${ROCGRAPH_BINARY_DIR}/fatbin.ld")

add_library(rocgraph::rocgraph ALIAS rocgraph)

################################################################################
# - include paths --------------------------------------------------------------

target_include_directories(rocgraph
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "${HIPToolkit_INCLUDE_DIRS}"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

################################################################################

target_link_libraries(rocgraph
    PUBLIC
        hip::toolkit
        roc::rocblas
        roc::rocsparse
        roc::rocrand
        roc::rocsolver
        $<BUILD_LOCAL_INTERFACE:ROCM::toolkit>

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
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               HIP_STANDARD                       17
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
