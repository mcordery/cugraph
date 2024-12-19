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

#pragma once

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/tuple.h>

namespace cugraph {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_erdos_renyi_graph_edgelist_gnp(raft::handle_t const& handle,
                                        vertex_t num_vertices,
                                        float p,
                                        vertex_t base_vertex_id,
                                        uint64_t seed)
{
  // NOTE:
  // https://networkx.org/documentation/stable/_modules/networkx/generators/random_graphs.html#fast_gnp_random_graph
  //   identifies a faster algorithm that I think would be very efficient on the GPU.  I believe we
  //   could just compute lr/lp in that code for a batch of values, use prefix sums to generate edge
  //   ids and then convert the generated values to a batch of edges.
  CUGRAPH_EXPECTS(num_vertices < std::numeric_limits<int32_t>::max(),
                  "Implementation cannot support specified value");

  size_t max_num_edges = static_cast<size_t>(num_vertices) * num_vertices;

  auto generate_random_value = [seed] __device__(size_t index) -> float {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist(0.0, 1.0);
    rng.discard(index);
    return dist(rng);
  };

  size_t count = thrust::count_if(handle.get_thrust_policy(),
                                  thrust::make_counting_iterator<size_t>(0),
                                  thrust::make_counting_iterator<size_t>(max_num_edges),
                                  [generate_random_value, p] __device__(size_t index) {
                                    return generate_random_value(index) < p;
                                  });

  rmm::device_uvector<vertex_t> src_v(count, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(count, handle.get_stream());

  thrust::copy_if(handle.get_thrust_policy(),
                  thrust::make_counting_iterator<size_t>(0),
                  thrust::make_counting_iterator<size_t>(max_num_edges),
                  thrust::make_transform_output_iterator(
                    thrust::make_zip_iterator(src_v.begin(), dst_v.begin()),
                    [num_vertices] __device__(size_t index) -> thrust::tuple<vertex_t, vertex_t> {
                      return thrust::make_tuple(static_cast<vertex_t>(index / num_vertices),
                                                static_cast<vertex_t>(index % num_vertices));
                    }),
                  [generate_random_value, p] __device__(size_t index) {
                    return generate_random_value(index) < p;
                  });

  return std::make_tuple(std::move(src_v), std::move(dst_v));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_erdos_renyi_graph_edgelist_gnm(raft::handle_t const& handle,
                                        vertex_t num_vertices,
                                        size_t m,
                                        vertex_t base_vertex_id,
                                        uint64_t seed)
{
  CUGRAPH_FAIL("Not implemented");

  // To implement:
  //    Use sampling function to select `m` unique edge ids from the
  //    (num_vertices ^ 2) possible edges.  Convert these to vertex
  //    ids.
}

}  // namespace cugraph
