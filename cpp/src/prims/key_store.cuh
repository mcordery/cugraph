/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/sort.h>

#include <hipco/static_set.cuh>

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>

namespace cugraph {

namespace detail {

using hipco_storage_type = hipco::storage<1>;  ///< hipco window storage type

template <typename KeyIterator>
struct key_binary_search_contains_op_t {
  using key_type = typename thrust::iterator_traits<KeyIterator>::value_type;

  KeyIterator store_key_first{};
  KeyIterator store_key_last{};

  __device__ bool operator()(key_type key) const
  {
    return thrust::binary_search(thrust::seq, store_key_first, store_key_last, key);
  }
};

template <typename ViewType>
struct key_binary_search_store_device_view_t {
  using key_type = typename ViewType::key_type;

  static_assert(ViewType::binary_search);

  __host__ key_binary_search_store_device_view_t(ViewType view)
    : store_key_first(view.store_key_first()), store_key_last(view.store_key_last())
  {
  }

  __device__ bool contains(key_type key) const
  {
    return thrust::binary_search(thrust::seq, store_key_first, store_key_last, key);
  }

  typename ViewType::key_iterator store_key_first{};
  typename ViewType::key_iterator store_key_last{};
};

template <typename ViewType>
struct key_hipco_store_contains_device_view_t {
  using key_type                   = typename ViewType::key_type;
  using hipco_store_device_ref_type = typename ViewType::hipco_set_type::ref_type<hipco::contains_tag>;

  static_assert(!ViewType::binary_search);

  __host__ key_hipco_store_contains_device_view_t(ViewType view)
    : hipco_store_device_ref(view.hipco_store_contains_device_ref())
  {
  }

  __device__ bool contains(key_type key) const { return hipco_store_device_ref.contains(key); }

  hipco_store_device_ref_type hipco_store_device_ref{};
};

template <typename ViewType>
struct key_hipco_store_insert_device_view_t {
  using key_type                   = typename ViewType::key_type;
  using hipco_store_device_ref_type = typename ViewType::hipco_set_type::ref_type<hipco::insert_tag>;

  static_assert(!ViewType::binary_search);

  __host__ key_hipco_store_insert_device_view_t(ViewType view)
    : hipco_store_device_ref(view.hipco_store_insert_device_ref())
  {
  }

  __device__ void insert(key_type key) { hipco_store_device_ref.insert(key); }

  hipco_store_device_ref_type hipco_store_device_ref{};
};

template <typename KeyIterator>
class key_binary_search_store_view_t {
 public:
  using key_type     = std::remove_cv_t<typename thrust::iterator_traits<KeyIterator>::value_type>;
  using key_iterator = KeyIterator;

  static constexpr bool binary_search = true;

  key_binary_search_store_view_t(KeyIterator key_first, KeyIterator key_last)
    : store_key_first_(key_first), store_key_last_(key_last)
  {
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void contains(QueryKeyIterator key_first,
                QueryKeyIterator key_last,
                ResultValueIterator value_first,
                rmm::cuda_stream_view stream) const
  {
    thrust::transform(
      rmm::exec_policy(stream),
      key_first,
      key_last,
      value_first,
      key_binary_search_contains_op_t<KeyIterator>{store_key_first_, store_key_last_});
  }

  KeyIterator store_key_first() const { return store_key_first_; }

  KeyIterator store_key_last() const { return store_key_last_; }

 private:
  KeyIterator store_key_first_{};
  KeyIterator store_key_last_{};
};

template <typename key_t>
class key_hipco_store_view_t {
 public:
  using key_type = key_t;

  static constexpr bool binary_search = false;

  using hipco_set_type =
    hipco::static_set<key_t,
                     hipco::extent<std::size_t>,
                     hip::thread_scope_device,
                     thrust::equal_to<key_t>,
                     hipco::linear_probing<1,  // CG size
                                          hipco::murmurhash3_32<key_t>>,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<std::byte>>,
                     hipco_storage_type>;

  key_hipco_store_view_t(hipco_set_type const* store) : hipco_store_(store) {}

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void contains(QueryKeyIterator key_first,
                QueryKeyIterator key_last,
                ResultValueIterator value_first,
                rmm::cuda_stream_view stream) const
  {
    hipco_store_->contains(key_first, key_last, value_first, stream);
  }

  auto hipco_store_contains_device_ref() const { return hipco_store_->ref(hipco::contains); }

  auto hipco_store_insert_device_ref() const { return hipco_store_->ref(hipco::insert); }

  key_t invalid_key() const { return hipco_store_->get_empty_key_sentinel(); }

 private:
  hipco_set_type const* hipco_store_{};
};

template <typename key_t>
class key_binary_search_store_t {
 public:
  using key_type = key_t;

  key_binary_search_store_t(rmm::cuda_stream_view stream) : store_keys_(0, stream) {}

  template <typename KeyIterator>
  key_binary_search_store_t(
    KeyIterator key_first,
    KeyIterator key_last,
    bool key_sorted /* if set to true, assume that the input data is sorted and skip sorting (which
                       is necessary for binary-search) */
    ,
    rmm::cuda_stream_view stream)
    : store_keys_(static_cast<size_t>(thrust::distance(key_first, key_last)), stream)
  {
    thrust::copy(rmm::exec_policy(stream), key_first, key_last, store_keys_.begin());
    if (!key_sorted) {
      thrust::sort(rmm::exec_policy(stream), store_keys_.begin(), store_keys_.end());
    }
  }

  key_binary_search_store_t(
    rmm::device_uvector<key_t>&& keys,
    bool key_sorted /* if set to true, assume that the input data is sorted and skip sorting (which
                       is necessary for binary-search) */
    ,
    rmm::cuda_stream_view stream)
    : store_keys_(std::move(keys))
  {
    if (!key_sorted) {
      thrust::sort(rmm::exec_policy(stream), store_keys_.begin(), store_keys_.end());
    }
  }

  auto release(rmm::cuda_stream_view stream)
  {
    auto tmp_store_keys = std::move(store_keys_);
    store_keys_         = rmm::device_uvector<key_t>(0, stream);
    return tmp_store_keys;
  }

  key_t const* store_key_first() const { return store_keys_.cbegin(); }

  key_t const* store_key_last() const { return store_keys_.cend(); }

  size_t size() const { return store_keys_.size(); }

  size_t capacity() const { return store_keys_.size(); }

 private:
  rmm::device_uvector<key_t> store_keys_;
};

template <typename key_t>
class key_hipco_store_t {
 public:
  using key_type = key_t;

  using hipco_set_type =
    hipco::static_set<key_t,
                     hipco::extent<std::size_t>,
                     hip::thread_scope_device,
                     thrust::equal_to<key_t>,
                     hipco::linear_probing<1,  // CG size
                                          hipco::murmurhash3_32<key_t>>,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<std::byte>>,
                     hipco_storage_type>;

  key_hipco_store_t(rmm::cuda_stream_view stream) {}

  key_hipco_store_t(size_t capacity, key_t invalid_key, rmm::cuda_stream_view stream)
  {
    allocate(capacity, invalid_key, stream);
    capacity_ = capacity;
    size_     = 0;
  }

  template <typename KeyIterator>
  key_hipco_store_t(KeyIterator key_first,
                   KeyIterator key_last,
                   key_t invalid_key,
                   rmm::cuda_stream_view stream)
  {
    auto num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
    allocate(num_keys, invalid_key, stream);
    capacity_ = num_keys;
    size_     = 0;

    insert(key_first, key_last, stream);
  }

  template <typename KeyIterator>
  void insert(KeyIterator key_first, KeyIterator key_last, rmm::cuda_stream_view stream)
  {
    auto num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
    if (num_keys == 0) return;

    size_ += hipco_store_->insert(key_first, key_last, stream.value());
  }

  template <typename KeyIterator, typename StencilIterator, typename PredOp>
  void insert_if(KeyIterator key_first,
                 KeyIterator key_last,
                 StencilIterator stencil_first,
                 PredOp pred_op,
                 rmm::cuda_stream_view stream)
  {
    auto num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
    if (num_keys == 0) return;

    size_ += hipco_store_->insert_if(key_first, key_last, stencil_first, pred_op, stream.value());
  }

  auto release(rmm::cuda_stream_view stream)
  {
    rmm::device_uvector<key_t> keys(size(), stream);
    auto last = hipco_store_->retrieve_all(keys.begin(), stream.value());
    keys.resize(thrust::distance(keys.begin(), last), stream);
    keys.shrink_to_fit(stream);
    allocate(0, invalid_key(), stream);
    capacity_ = 0;
    size_     = 0;
    return keys;
  }

  hipco_set_type const* hipco_store_ptr() const { return hipco_store_.get(); }

  key_t invalid_key() const { return hipco_store_->empty_key_sentinel(); }

  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

 private:
  void allocate(size_t num_keys, key_t invalid_key, rmm::cuda_stream_view stream)
  {
    double constexpr load_factor = 0.7;
    auto hipco_size               = std::max(
      static_cast<size_t>(static_cast<double>(num_keys) / load_factor),
      static_cast<size_t>(num_keys) + 1);  // hipco::static_map requires at least one empty slot

    auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(
      rmm::mr::polymorphic_allocator<std::byte>(rmm::mr::get_current_device_resource()), stream);
    hipco_store_ =
      std::make_unique<hipco_set_type>(hipco_size,
                                      hipco::empty_key<key_t>{invalid_key},
                                      thrust::equal_to<key_t>{},
                                      hipco::linear_probing<1,  // CG size
                                                           hipco::murmurhash3_32<key_t>>{},
                                      hipco::thread_scope_device,
                                      hipco_storage_type{},
                                      stream_adapter,
                                      stream.value());
  }

  std::unique_ptr<hipco_set_type> hipco_store_{nullptr};

  size_t capacity_{0};
  size_t size_{0};  // caching as hipco_store_->size() is expensive (this scans the entire slots to
                    // handle user inserts through a device reference
};

}  // namespace detail

/* a class to store keys, the actual storage can either be implemented based on binary tree (when
 * use_binary_search == true) or hash-table (cuCollection, when use_binary_search = false) */
template <typename key_t, bool use_binary_search = true>
class key_store_t {
 public:
  using key_type = key_t;

  static_assert(std::is_arithmetic_v<key_t>);

  key_store_t(rmm::cuda_stream_view stream) : store_(stream) {}

  /* when use_binary_search = false */
  template <bool binary_search = use_binary_search>
  key_store_t(
    size_t capacity /* one can expect good performance till the capacity, the actual underlying
                       capacity can be larger (for performance & correctness reasons) */
    ,
    key_t invalid_key /* invalid key shouldn't appear in any *iter in [key_first, key_last) */,
    rmm::cuda_stream_view stream,
    std::enable_if_t<!binary_search, int32_t> = 0)
    : store_(capacity, invalid_key, stream)
  {
  }

  /* when use_binary_search = true */
  template <typename KeyIterator, bool binary_search = use_binary_search>
  key_store_t(KeyIterator key_first,
              KeyIterator key_last,
              bool key_sorted /* if set to true, assume that the input data is sorted and skip
                                 sorting (which is necessary for binary-search) */
              ,
              rmm::cuda_stream_view stream,
              std::enable_if_t<binary_search, int32_t> = 0)
    : store_(key_first, key_last, key_sorted, stream)
  {
  }

  /* when use_binary_search = false */
  template <typename KeyIterator, bool binary_search = use_binary_search>
  key_store_t(
    KeyIterator key_first,
    KeyIterator key_last,
    key_t invalid_key /* invalid key shouldn't appear in any *iter in [key_first, key_last) */,
    rmm::cuda_stream_view stream,
    std::enable_if_t<!binary_search, int32_t> = 0)
    : store_(key_first, key_last, invalid_key, stream)
  {
  }

  /* when use_binary_search = true */
  template <bool binary_search = use_binary_search>
  key_store_t(rmm::device_uvector<key_t>&& keys,
              bool key_sorted /* if set to true, assume that the input data is sorted and skip
                                 sorting (which is necessary for binary-search) */
              ,
              rmm::cuda_stream_view stream,
              std::enable_if_t<binary_search, int32_t> = 0)
    : store_(std::move(keys), key_sorted, stream)
  {
  }

  /* when use binary_search = false, this requires that the capacity is large enough */
  template <typename KeyIterator, bool binary_search = use_binary_search>
  std::enable_if_t<!binary_search, void> insert(KeyIterator key_first,
                                                KeyIterator key_last,
                                                rmm::cuda_stream_view stream)
  {
    store_.insert(key_first, key_last, stream);
  }

  /* when use binary_search = false, this requires that the capacity is large enough */
  template <typename KeyIterator,
            typename StencilIterator,
            typename PredOp,
            bool binary_search = use_binary_search>
  std::enable_if_t<!binary_search, void> insert_if(KeyIterator key_first,
                                                   KeyIterator key_last,
                                                   StencilIterator stencil_first,
                                                   PredOp pred_op,
                                                   rmm::cuda_stream_view stream)
  {
    store_.insert_if(key_first, key_last, stencil_first, pred_op, stream);
  }

  // key_store_t becomes empty after release
  auto release(rmm::cuda_stream_view stream) { return store_.release(stream); }

  auto view() const
  {
    if constexpr (use_binary_search) {
      return detail::key_binary_search_store_view_t(store_.store_key_first(),
                                                    store_.store_key_last());
    } else {
      return detail::key_hipco_store_view_t<key_t>(store_.hipco_store_ptr());
    }
  }

  template <bool binary_search = use_binary_search>
  std::enable_if_t<!binary_search, key_t> invalid_key() const
  {
    return store_.invalid_key();
  }

  size_t size() const { return store_.size(); }

  size_t capacity() const { return store_.capacity(); }

 private:
  std::conditional_t<use_binary_search,
                     detail::key_binary_search_store_t<key_t>,
                     detail::key_hipco_store_t<key_t>>
    store_;
};

}  // namespace cugraph
