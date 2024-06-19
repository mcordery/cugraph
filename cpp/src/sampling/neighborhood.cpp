/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "utilities/cugraph_ops_utils.hpp"

#include <cugraph-ops/graph/sampling.hpp>
#include <cugraph/algorithms.hpp>

#include <raft/random/rng_state.hpp>

#include <type_traits>

namespace cugraph {

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>>
sample_neighbors_adjacency_list(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
                                vertex_t const* ptr_d_start,
                                size_t num_start_vertices,
                                size_t sampling_size,
                                ops::graph::SamplingAlgoT sampling_algo)
{
  using base_vertex_t = std::decay_t<vertex_t>;
  using base_edge_t   = std::decay_t<edge_t>;
  static_assert(std::is_same_v<base_vertex_t, base_edge_t>,
                "cugraph-ops sampling not yet implemented for different node and edge types");

  const auto ops_graph = detail::get_graph(graph_view);
  return ops::graph::uniform_sample_csc(rng_state,
                                        ops_graph,
                                        ptr_d_start,
                                        num_start_vertices,
                                        sampling_size,
                                        sampling_algo,
                                        ops_graph.dst_max_in_degree,
                                        handle.get_stream());
}

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> sample_neighbors_edgelist(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  vertex_t const* ptr_d_start,
  size_t num_start_vertices,
  size_t sampling_size,
  ops::graph::SamplingAlgoT sampling_algo)
{
  using base_vertex_t = std::decay_t<vertex_t>;
  using base_edge_t   = std::decay_t<edge_t>;
  static_assert(std::is_same_v<base_vertex_t, base_edge_t>,
                "cugraph-ops sampling not yet implemented for different node and edge types");

  const auto ops_graph = detail::get_graph(graph_view);
  return ops::graph::uniform_sample_coo(rng_state,
                                        ops_graph,
                                        ptr_d_start,
                                        num_start_vertices,
                                        sampling_size,
                                        sampling_algo,
                                        ops_graph.dst_max_in_degree,
                                        handle.get_stream());
}

// template explicit instantiation directives (EIDir's):
//
// CSR SG FP32{
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
sample_neighbors_adjacency_list(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                graph_view_t<int32_t, int32_t, false, false> const& gview,
                                int32_t const* ptr_d_start,
                                size_t num_start_vertices,
                                size_t sampling_size,
                                ops::graph::SamplingAlgoT sampling_algo);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
sample_neighbors_adjacency_list(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                graph_view_t<int64_t, int64_t, false, false> const& gview,
                                int64_t const* ptr_d_start,
                                size_t num_start_vertices,
                                size_t sampling_size,
                                ops::graph::SamplingAlgoT sampling_algo);
//}
//
// COO SG FP32{
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
sample_neighbors_edgelist(raft::handle_t const& handle,
                          raft::random::RngState& rng_state,
                          graph_view_t<int32_t, int32_t, false, false> const& gview,
                          int32_t const* ptr_d_start,
                          size_t num_start_vertices,
                          size_t sampling_size,
                          ops::graph::SamplingAlgoT sampling_algo);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
sample_neighbors_edgelist(raft::handle_t const& handle,
                          raft::random::RngState& rng_state,
                          graph_view_t<int64_t, int64_t, false, false> const& gview,
                          int64_t const* ptr_d_start,
                          size_t num_start_vertices,
                          size_t sampling_size,
                          ops::graph::SamplingAlgoT sampling_algo);
//}

}  // namespace cugraph