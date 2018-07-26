
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <utility>
#include "epoch_reclaimer.h"
#include "hash_table_utils.h"


namespace HT {

namespace CPU {


namespace sequential {

namespace {
// A general robin hood hash table with backward shift deletion.
template <class Data, class K, class H, class Helpers, class Allocator = std::allocator<Data>> class robin_hood_hashtable {
private:
  // Stores an arbritray number of bits per entry
  // compressing when needed.
  class bit_vector {
  private:
    std::size_t m_capacity, m_num_bits;
    std::uint8_t *m_data;

  public:
    bit_vector(std::size_t capacity = 32, std::size_t num_bits = 1)
        : m_capacity(capacity * num_bits), m_num_bits(num_bits),
          m_data(new std::uint8_t[m_capacity]) {
      assert(m_capacity > 0);
    }

    bit_vector(const bit_vector & rhs):
      m_capacity(rhs.m_capacity),
      m_num_bits(rhs.m_num_bits),
      m_data(new std::uint8_t[rhs.m_capacity]) {
      for(std::size_t i = 0; i < m_capacity; i++) {
        m_data[i] = rhs.m_data[i];
      }
    }

    bit_vector(bit_vector && rhs):
      m_capacity(rhs.m_capacity),
      m_num_bits(rhs.m_num_bits),
      m_data(rhs.m_data) {
      rhs.m_data = nullptr;
    }

    bit_vector & operator=(const bit_vector & rhs)
    {
      if(this != &rhs) {
        delete[] m_data;
        m_capacity = rhs.m_capacity,
        m_num_bits = rhs.m_num_bits;
        m_data = new std::uint8_t[rhs.m_capacity];
        for(std::size_t i = 0; i < m_capacity; i++){
          m_data[i] = rhs.m_data[i];
        }
      }
      return *this;
    }

    bit_vector & operator=(bit_vector && rhs)
    {
      if(this != &rhs) {
        delete[] m_data;
        m_capacity = rhs.m_capacity,
        m_num_bits = rhs.m_num_bits;
        m_data = rhs.m_data;
        rhs.m_data = nullptr;
      }
      return *this;
    }

    ~bit_vector() {
      delete [] m_data;
    }

    bool get(std::size_t position, std::size_t specific_bit = 0) {
      assert(position < m_capacity and "Position requested exceeded capacity.");
      assert(specific_bit < m_num_bits and
             "Bit requested is great than number of bits stored.");
      std::size_t adjusted_position = position * m_num_bits;
      std::size_t block = adjusted_position / sizeof(std::uint8_t);
      std::size_t bit = adjusted_position % sizeof(std::uint8_t);
      std::uint8_t current_value = m_data[block];
      return ((current_value >> (bit + specific_bit)) & 1) == 1;
    }

    bool set(std::size_t position, bool value, std::size_t specific_bit = 0) {
      assert(position < m_capacity and
             "Position being changed exceeded capacity.");
      assert(specific_bit < m_num_bits and
             "Bit requested is great than number of bits stored.");
      std::size_t adjusted_position = position * m_num_bits;
      std::size_t block = adjusted_position / sizeof(std::uint8_t);
      std::size_t bit = adjusted_position % sizeof(std::uint8_t);
      std::uint8_t current_value = m_data[block];
      if (value) {
        m_data[block] |= 1 << (bit + specific_bit);
      } else {
        m_data[block] &= ~(1 << (bit + specific_bit));
      }
      return ((current_value >> (bit + specific_bit)) & 1) == 1;
    }
  };
  
  std::size_t m_count, m_capacity, m_capacity_mask, m_probe_threshold;
  const double m_load_factor_threshold; 
  Data *m_data;
  bit_vector m_array_info;
  H m_hash;

  static const std::size_t FREE_BIT = 0;
  static const std::size_t NUM_BITS = 1;

  void resize() {
    std::size_t count = m_count;
    robin_hood_hashtable<Data, K, H, Helpers, Allocator> new_rh(this->m_capacity * 2, this->m_load_factor_threshold);
    for(auto it = this->begin(); it != this->end(); it++){
      new_rh.insert(*it);
    }
    (*this) = std::move(new_rh);
  }

public:
  class iterator {
  public:
    typedef iterator self_type;
    typedef Data value_type;
    typedef Data &reference;
    typedef Data *pointer;
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef std::ptrdiff_t difference_type;
    iterator(robin_hood_hashtable *table, pointer ptr, std::size_t bucket_idx)
        : m_table(table), m_pointer(ptr), m_bucket_idx(bucket_idx) {}
    self_type operator++() {
      self_type i = *this;
      for(; ++m_bucket_idx < m_table->m_capacity; ++m_pointer) {
        if(!m_table->m_array_info.get(m_bucket_idx, robin_hood_hashtable::FREE_BIT)) { break; }
      }
      ++m_pointer;
      return i;
    }
    self_type operator++(int _) {
      for(; ++m_bucket_idx < m_table->m_capacity; ++m_pointer) {
        if(!m_table->m_array_info.get(m_bucket_idx, robin_hood_hashtable::FREE_BIT)) { break; }
      }
      ++m_pointer;
      return *this;
    }
    self_type operator--() {
      self_type i = *this;
      for(;m_table->m_array_info.get(--m_bucket_idx, robin_hood_hashtable::FREE_BIT); --m_pointer) {}
      --m_pointer;
      return i;
    }
    self_type operator--(int _) {
      for(;m_table->m_array_info.get(--m_bucket_idx, robin_hood_hashtable::FREE_BIT); --m_pointer) {}
      --m_pointer;
      return *this;
    }
    reference operator*() { return *m_pointer; }
    pointer operator->() { return m_pointer; }
    bool operator==(const self_type &rhs) { return m_pointer == rhs.m_pointer; }
    bool operator!=(const self_type &rhs) { return m_pointer != rhs.m_pointer; }

  private:
    robin_hood_hashtable *m_table;
    pointer m_pointer;
    std::size_t m_bucket_idx;
    friend class robin_hood_hashtable;
  };

  friend class iterator;

  robin_hood_hashtable(std::size_t capacity = 32, double load_factor_threshold = 0.75)
      : m_count(0),
        m_capacity(utils::nearest_power_of_two(capacity)),
        m_capacity_mask(m_capacity - 1),
        m_load_factor_threshold(load_factor_threshold),
        m_data(static_cast<Data *>(calloc(m_capacity, sizeof(Data)))),
        m_array_info(m_capacity, NUM_BITS) {
    assert(m_capacity > 0);
    for (std::size_t i = 0; i < m_capacity; i++) {
      m_array_info.set(i, true, FREE_BIT);
    }
  }

  robin_hood_hashtable(const robin_hood_hashtable &rhs):
    m_count(rhs.m_count),
    m_capacity(rhs.m_capacity),
    m_capacity_mask(m_capacity - 1),
    m_load_factor_threshold(rhs.m_load_factor_threshold),
    m_data(static_cast<Data *>(calloc(m_capacity, sizeof(Data)))),
    m_array_info(rhs.m_array_info)
  {
    for(std::size_t i = 0; i < m_capacity; i++) {
      this->m_data[i] = rhs.m_data[i];
    }
  }

  robin_hood_hashtable(robin_hood_hashtable &&rhs):
    m_count(rhs.m_count),
    m_capacity(rhs.m_capacity),
    m_capacity_mask(m_capacity - 1),
    m_load_factor_threshold(rhs.m_load_factor_threshold),
    m_data(rhs.m_data),
    m_array_info(std::move(rhs.m_array_info))
  {
    rhs.m_count = 0;
    rhs.m_capacity = 0;
    rhs.m_data = nullptr;
  }

  robin_hood_hashtable & operator=(const robin_hood_hashtable &rhs)
  {
    if(this != &rhs) {
      for (std::size_t i = 0; i < m_capacity; i++) {
        if (!m_array_info.get(i, FREE_BIT)) {
          m_data[i].~Data();
        }
      }
      free(m_data);
      m_count = rhs.m_count;
      m_capacity = rhs.m_capacity;
      m_capacity_mask = rhs.m_capacity_mask;
      m_data = static_cast<Data *>(calloc(m_capacity, sizeof(Data)));
      m_array_info = rhs.m_array_info;
      for(std::size_t i = 0; i < m_capacity; i++) {
        this->m_data[i] = rhs.m_data[i];
      }
    }
    return *this;
  }

  robin_hood_hashtable& operator=(robin_hood_hashtable &&rhs)
  {
    if(this != &rhs) {
      for (std::size_t i = 0; i < m_capacity; i++) {
        if (!m_array_info.get(i, FREE_BIT)) {
          m_data[i].~Data();
        }
      }
      free(m_data);
      m_count = rhs.m_count;
      m_capacity = rhs.m_capacity;
      m_capacity_mask = rhs.m_capacity_mask;
      m_data = rhs.m_data;
      m_array_info = std::move(rhs.m_array_info);
      rhs.m_count = 0;
      rhs.m_capacity = 0;
      rhs.m_data = nullptr;
    }
    return *this;
  }

  ~robin_hood_hashtable() {
    for (std::size_t i = 0; i < m_capacity; i++) {
      if (!m_array_info.get(i, FREE_BIT)) {
        m_data[i].~Data();
      }
    }
    free(m_data);
  }

  iterator begin() {
    // Find first element
    for (std::size_t i = 0; i < m_capacity; i++) {
      if (!m_array_info.get(i, FREE_BIT)) {
        return iterator(this, &m_data[i], i);
      }
    }
    return this->end();
  }

  iterator end() { return iterator(this, &m_data[m_capacity], m_capacity); }

  iterator find(const K &key) {
    std::size_t current_bucket = m_hash(key);
    for (std::size_t search_dist = 0;; search_dist++, current_bucket++) {
      current_bucket &= m_capacity_mask;

      if (m_array_info.get(current_bucket, FREE_BIT)) {
        return this->end();
      }

      Data &current_entry = m_data[current_bucket];
      const K current_key = Helpers::get_key(current_entry);
      if (key == current_key) {
        return iterator(this, &m_data[current_bucket], current_bucket);
      }
      std::size_t cur_home_slot = m_hash(current_key) & m_capacity_mask;
      std::size_t cur_distance = utils::distance_from_slot(m_capacity, cur_home_slot, current_bucket);
      if (search_dist > cur_distance) {
        return this->end();
      }
    }
  }

  std::pair<iterator, bool> insert(const Data &data) {
    Data active_data = data;
    const K inserting_key = Helpers::get_key(data);
    if((double(m_count) / double(m_capacity)) > m_load_factor_threshold) {
      this->resize();
    }
    iterator insertion_bucket = this->end();
    std::size_t current_bucket = m_hash(inserting_key);
    for (std::size_t search_dist = 0;; search_dist++, current_bucket++) {
      current_bucket &= m_capacity_mask;

      if (m_array_info.get(current_bucket, FREE_BIT)) {
        m_array_info.set(current_bucket, false, FREE_BIT);
        new (m_data + current_bucket) Data(active_data);
        m_count++;
        return std::make_pair(
            insertion_bucket == this->end()
                ? iterator(this, &m_data[current_bucket], current_bucket)
                : insertion_bucket,
            true);
      }

      Data &current_data = m_data[current_bucket];
      const K current_key = Helpers::get_key(current_data);
      if (current_key == inserting_key) {
        // Update key/value pairing? Nah
        return std::make_pair(iterator(this, &m_data[current_bucket], current_bucket),
                              false);
      }
      std::size_t cur_home_slot = m_hash(current_key) & m_capacity_mask;
      std::size_t cur_distance =
          utils::distance_from_slot(m_capacity, cur_home_slot, current_bucket);
      if (cur_distance < search_dist) {
        Data temp = m_data[current_bucket];
        m_data[current_bucket] = active_data;
        active_data = temp;
        search_dist = cur_distance;
        // memcpy(m_data + current_bucket, &active_data, sizeof(Data));
        if (insertion_bucket == this->end()) {
          insertion_bucket = iterator(this, &m_data[current_bucket], current_bucket);
        }
      }
    }
  }

  template <class Func>
  std::pair<iterator, bool> upsert(const Data &data, const Func &func) {
    iterator it = this->find(Helpers::get_key(data));
    // Update the value
    if(it != this->end()) {
      Data new_data = func(*it);
      assert(Helpers::get_key(data) == Helpers::get_key(new_data));
      (*it) = new_data;
      return std::make_pair(it, true);
    } else {
      return std::make_pair(this->insert(data).first, false);
    }
  }

  void erase(const K& key) {
    iterator pos = this->find(key);
    if (pos != this->end()) {
      // m_data[pos.m_bucket_idx].~Data();
      std::size_t src_idx = pos.m_bucket_idx;
      std::size_t dest_idx = src_idx + 1;
      while(true){
        src_idx &= m_capacity_mask;
        dest_idx &= m_capacity_mask;

        if(m_array_info.get(dest_idx, FREE_BIT)){
          break;
        }
        const Data &dest_data = m_data[dest_idx];
        const K dest_key = Helpers::get_key(dest_data);
        std::size_t dest_home_slot = m_hash(dest_key) & m_capacity_mask;
        std::size_t dest_distance =
          utils::distance_from_slot(m_capacity, dest_home_slot, dest_idx);
        if(dest_distance == 0){
          break;
        }
        m_data[src_idx++] = m_data[dest_idx++];
      }
      m_data[src_idx].~Data();
      m_array_info.set(src_idx, true, FREE_BIT);
      m_count--;
    }
  }

  std::size_t size(){
    return m_count;
  }

  bool contains(const K& key) { return this->find(key) != this->end(); }
};

template <class K, class Data>
class robin_hood_unordered_set_helper {
public:
  static K get_key(const Data & data){
    return data;
  }
};

template <class K, class Data>
class robin_hood_unordered_map_helper {
public:
  static K get_key(const Data & data){
    return data.first;
  }
};
}

template <class K, class H = std::hash<K>>
using robin_hood_unordered_set = robin_hood_hashtable<K, K, H, robin_hood_unordered_set_helper<K, K>>;

template <class K, class V, class H = std::hash<K>>
using robin_hood_unordered_map = robin_hood_hashtable<std::pair<K, V>, K, H, robin_hood_unordered_map_helper<K, std::pair<K, V>>>;
}

namespace concurrent {

enum class ClaimOperationResult { Claimed, GrowthNeeded, GrowthOngoing};

template <class K, class V, class KT = utils::KeyTraits<K>, class VT = utils::ValueTraits<V>>
class lf_linear_probing {
public:
  struct Bucket {
    std::atomic<K> key;
    std::atomic<V> value;
  };

  const std::size_t m_size, m_size_mask, m_probe_threshold, m_buckets_per_copy_id;
  std::atomic_size_t m_occupied_buckets, m_tombstoned_buckets, m_bucket_copy_id, m_buckets_copied;
  const double m_load_factor_threshold;
  std::mutex m_grow_lock;
  Bucket * m_buckets;
  std::atomic<lf_linear_probing *> m_next_table;
  
public:
  lf_linear_probing(std::size_t size, double load_factor_threshold = 0.7, std::size_t probe_theshold = 70):
    m_size(utils::nearest_power_of_two(std::max(std::size_t(32), size))),
    m_size_mask(m_size - 1),
    m_probe_threshold(probe_theshold),
    m_buckets_per_copy_id(16),
    m_occupied_buckets(std::size_t(0)),
    m_tombstoned_buckets(std::size_t(0)),
    m_bucket_copy_id(std::size_t(0)),
    m_buckets_copied(std::size_t(0)),
    m_load_factor_threshold(load_factor_threshold),
    m_buckets(new Bucket[m_size]),
    m_next_table(nullptr)
  {
    for(std::size_t i = 0; i < m_size; i++) {
      m_buckets[i].key.store(KT::NullKey, std::memory_order_relaxed);
      m_buckets[i].value.store(VT::NullValue, std::memory_order_relaxed);
    }
  }

  ~lf_linear_probing(){ delete[] m_buckets; }
  
  static Bucket * find(lf_linear_probing * table, const K &key) {
    Bucket * buckets = table->m_buckets;
    std::size_t size_mask = table->m_size_mask, probe_theshold = table->m_probe_threshold;
    std::size_t original_bucket = KT::hash(key) & size_mask;
    bool ran_once = false;
    for(std::size_t cur_bucket = original_bucket, num_probes = 0; true; cur_bucket++, num_probes++, ran_once = true){
      if(num_probes > probe_theshold) {
       return nullptr;
      }
      cur_bucket &= size_mask;
      if(ran_once and cur_bucket == original_bucket) {
        return nullptr;
      }
      const K cur_key = buckets[cur_bucket].key.load(std::memory_order_relaxed);
      if(cur_key == KT::NullKey) {
        return nullptr;
      }
      if(cur_key == key) {
        return buckets + cur_bucket;
      }
    }
  }
  
  static std::pair<ClaimOperationResult, Bucket*> claim_bucket(lf_linear_probing * table, const K &key) {
    assert(key != KT::NullKey and key != KT::DeadKey);
    const std::size_t size = table->m_size;
    const double load_factor_threshold = table->m_load_factor_threshold;
    const std::size_t occupied_buckets = table->m_occupied_buckets.load(std::memory_order_relaxed);
    ClaimOperationResult result = ClaimOperationResult::Claimed;
    // Calculate load factor and see if it violates threshold.
    if((double(occupied_buckets) / double(size)) > load_factor_threshold) {
      result = ClaimOperationResult::GrowthNeeded;
    }
    const std::size_t size_mask = table->m_size_mask, probe_theshold = table->m_probe_threshold;
    Bucket * buckets = table->m_buckets;
    const std::size_t original_bucket = KT::hash(key) & size_mask;
    bool ran_once = false;
    std::size_t cur_bucket = original_bucket;
    for(std::size_t num_probes = 0; true; cur_bucket++, num_probes++, ran_once = true){
      if(num_probes > probe_theshold) {
        return std::make_pair(ClaimOperationResult::GrowthNeeded, nullptr);
      }
      cur_bucket &= size_mask;
      if(ran_once and cur_bucket == original_bucket) {
        return std::make_pair(ClaimOperationResult::GrowthNeeded, nullptr);
      }
      K cur_key = buckets[cur_bucket].key.load(std::memory_order_relaxed);
    loadBegin:
      if(cur_key == KT::NullKey) {
        if(!buckets[cur_bucket].key.compare_exchange_weak(cur_key, key, std::memory_order_relaxed, std::memory_order_relaxed)) {
          goto loadBegin;
        }
        table->m_occupied_buckets.fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(result, buckets + cur_bucket);
      }
      if(cur_key == KT::DeadKey) {
        return std::make_pair(ClaimOperationResult::GrowthOngoing, nullptr);
      }
      if(cur_key == key) {
        return std::make_pair(result, buckets + cur_bucket);
      }
    }
  }
};

template <template<class, class, class, class> class T, class K, class V, class KT = utils::KeyTraits<K>, class VT = utils::ValueTraits<V>>
class table_manager {
private:
  typedef T<K, V, KT, VT> Table;
  typedef typename Table::Bucket Bucket;
  std::atomic<Table *> m_root;
  epoch_reclaimer m_reclaimer;

  static bool verify_copy(Table * source_table, Table * destination_table) {
    for(std::size_t i = 0; i < source_table->m_size; i++) {
      const K cur_key = source_table->m_buckets[i].key.load(std::memory_order_relaxed);
      const V cur_value = source_table->m_buckets[i].value.load(std::memory_order_relaxed);
      if(cur_key == KT::NullKey) {
        std::cout << "NULL KEY!!!" << std::endl;
        return false;
      } else if(cur_key != KT::DeadKey and cur_value != VT::RedirectValue) {
        std::cout << "NO REDIRECT VALUE!!!" << std::endl;
        return false;
      }
    }
    return true;
  }
  
  static Table * load_latest_table(Table * root) {
    Table * current = root;
    Table * next_table = current->m_next_table.load(std::memory_order_consume);
    while(next_table != nullptr) {
      current = next_table;
      next_table = current->m_next_table.load(std::memory_order_consume);
    }
    return current;
  }
  
  static Table * load_next_table(Table * current) {
    Table * next_table = current->m_next_table.load(std::memory_order_consume);
    while(next_table == nullptr) {
      next_table = current->m_next_table.load(std::memory_order_consume);
    }
    return next_table;
  }

  static Table *grow_table(Table * table) {
    Table *next = table->m_next_table.load(std::memory_order_consume);
    if(next == nullptr) {
      std::lock_guard<std::mutex> guard(table->m_grow_lock);
      next = table->m_next_table.load(std::memory_order_consume);
      if(next != nullptr) {
        return next;
      }
      // Now we need to figure out if we grow or not.
      std::size_t current_size = table->m_size;
      std::size_t new_table_size = current_size * 2;
      std::size_t num_tombstoned_buckets = table->m_tombstoned_buckets.load(std::memory_order_relaxed);
      // More than half are tombstones, same size please.
      if(num_tombstoned_buckets * 2 > table->m_size){
        new_table_size = current_size;
      } else if(double(num_tombstoned_buckets) / double(current_size) > table->m_load_factor_threshold){
        new_table_size = current_size / 2;
      }
      Table *new_table = new Table(std::max(new_table_size, std::size_t(32)));
      table->m_next_table.store(new_table, std::memory_order_release);
      return new_table;
    } else {
      return next;
    }
  }

  static bool try_move_value(Bucket *source_bucket, V source_value, Bucket *destination_bucket) {
    assert(source_bucket->key.load(std::memory_order_relaxed) == destination_bucket->key.load(std::memory_order_relaxed));
    assert(source_value != VT::NullValue or source_value != VT::Tombstone);
    V destination_value = destination_bucket->value.load(std::memory_order_relaxed);

    // Mark the bucket as being ready to copy.
    while(true) {
      if(source_value == VT::RedirectValue) {
        return false;
      }
      if(VT::is_primed(source_value)) {
        break;
      }
      if(source_bucket->value.compare_exchange_weak(source_value, VT::make_primed(source_value), std::memory_order_relaxed, std::memory_order_relaxed)) {
        break;
      }
    }
    source_value = VT::get_value(source_value);
    while(true) {
      // We're totally fucked. Someone is already copying the table we want to copy to...
      if(destination_value == VT::RedirectValue or VT::is_primed(destination_value)) {
        break;
      } else if(destination_value == VT::NullValue and destination_bucket->value.compare_exchange_weak(destination_value, source_value, std::memory_order_relaxed, std::memory_order_relaxed)) {
        // Empty value case.
        break;
      } else if(destination_value != VT::RedirectValue and destination_value != VT::NullValue) {
        // Value is copied by someone else.
        // Try to overwrite the old value with redirect
        break;
      }
    }
    // Now, back to value we wrote in.
    source_value = VT::make_primed(source_value);
    while(true) {
      if(source_bucket->value.compare_exchange_weak(source_value, VT::RedirectValue, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return true;
      }
      if(VT::get_value(source_value) == VT::RedirectValue) {
        return false;
      }
      source_value = VT::make_primed(source_value);
    }
  }

  static void copy_bucket(Table *source_table, Bucket * source_bucket, Table * target_table) {
    while(true) {
      K source_key = source_bucket->key.load(std::memory_order_relaxed);
      V raw_source_value = source_bucket->value.load(std::memory_order_relaxed);
      V source_value = VT::get_value(raw_source_value);
      // Determine if we should copy this value.
      if(source_key == KT::NullKey and source_bucket->key.compare_exchange_weak(source_key, KT::DeadKey, std::memory_order_relaxed, std::memory_order_relaxed)) {
        source_table->m_buckets_copied.fetch_add(1, std::memory_order_relaxed);
        return;
      } else if(source_key == KT::DeadKey) {
        return;
      } else if(source_value == VT::RedirectValue) { // Already copied value.
        return;
      } else if((source_value == VT::Tombstone or source_value == VT::NullValue) and source_bucket->value.compare_exchange_weak(raw_source_value, VT::RedirectValue, std::memory_order_relaxed, std::memory_order_relaxed)) {
        source_table->m_buckets_copied.fetch_add(1, std::memory_order_relaxed);
        return;
      } else {
        // Try claim new bucket in the destination table.
        std::pair<ClaimOperationResult, Bucket *> destination_result = Table::claim_bucket(target_table, source_key);
        ClaimOperationResult destination_op_result = destination_result.first;
        Bucket * destination_bucket = destination_result.second;
        if(destination_op_result == ClaimOperationResult::Claimed) {
          // Try copy the value over, check if the value is the same once copied, and then mark it with redirect.
          // Done! Next bucket please!
          if(try_move_value(source_bucket, raw_source_value, destination_bucket)) {
            source_table->m_buckets_copied.fetch_add(1, std::memory_order_relaxed);
          }
          return;
        } else if(destination_op_result == ClaimOperationResult::GrowthNeeded or destination_op_result == ClaimOperationResult::GrowthOngoing) {
          target_table = grow_table(target_table);
        }
      }
    }
  }

  static void copy_table(Table * source_table) {
    Table *destination_table = grow_table(source_table);
    Bucket *source_buckets = source_table->m_buckets;
    std::size_t buckets_per_copy_id = source_table->m_buckets_per_copy_id;
    std::size_t max_copy_id = source_table->m_size / source_table->m_buckets_per_copy_id;
    for(std::size_t current_block = source_table->m_bucket_copy_id.fetch_add(1, std::memory_order_relaxed); 
      current_block < max_copy_id; current_block = source_table->m_bucket_copy_id.fetch_add(1, std::memory_order_relaxed)) {
      std::size_t offset = current_block * buckets_per_copy_id;
      for(std::size_t i = 0; i < buckets_per_copy_id; i++) {
        std::size_t current_bucket = offset + i;
        copy_bucket(source_table, &source_buckets[current_bucket], destination_table);
      }
    }
  }

  void upgrade_table(epoch_reclaimer::epoch_guard &guard) {
    while(true) {
      Table *root = m_root.load(std::memory_order_consume);
      Table *current = root;
      std::size_t current_size = current->m_size;
      Table *begining = current, *end = nullptr;
      // While we still have the most up to date table...
      while(current->m_buckets_copied.load(std::memory_order_relaxed) == current_size) {
        Table *next = load_next_table(current);
        if(next == nullptr) {
          break;
        }
        end = current;
        current = next;
        current_size = current->m_size;
      }
      
      if(end == nullptr) {
        return;
      }

      if(m_root.compare_exchange_weak(root, current, std::memory_order_release, std::memory_order_consume)) {
        // Retire all beginning to end.
        while(begining != end) {
          guard.add_garbage([begining]() { verify_copy(begining, nullptr); delete begining; });
          begining = begining->m_next_table.load(std::memory_order_consume);
        }
        guard.add_garbage([end]() { verify_copy(end, nullptr); delete end; });
        return;
      }
    }
  }
  
public:
  table_manager(std::size_t initial_size = 32, std::size_t num_threads = std::thread::hardware_concurrency()):
    m_root(new Table(initial_size)),
    m_reclaimer(num_threads) {
      std::cout << VT::PrimedBit << " " << VT::NullValue << " " << VT::Tombstone << " " << VT::RedirectValue << std::endl;
      std::cout << VT::is_primed(VT::PrimedBit) << " " << VT::is_primed(VT::NullValue) << " " << VT::is_primed(VT::Tombstone) << " " << VT::is_primed(VT::RedirectValue) << std::endl;
    }
  ~table_manager() {
    delete m_root.load(std::memory_order_consume);
  }
  
  V find(const K &key, const std::size_t thread_id) {
    assert(key != KT::NullKey and key != KT::DeadKey);
    epoch_reclaimer::epoch_guard guard(m_reclaimer.get_epoch(thread_id));
    Table * table = m_root.load(std::memory_order_consume);
    // Deepest valid value.
    V deepest_value = VT::NullValue; // TODO: Should I be non-null?
    assert(table != nullptr);
    while(true) {
      Bucket * bucket = Table::find(table, key);
      if(bucket != nullptr) {
        V cur_value = VT::get_value(bucket->value.load(std::memory_order_relaxed));
        // Valid value eg. not a redirect or a null
        if(cur_value != VT::RedirectValue and cur_value != VT::NullValue) {
          deepest_value = cur_value;
        }
      }
      Table *next = table->m_next_table.load(std::memory_order_consume);
      if(next == nullptr) { // No other tables.
        return deepest_value;
      }
      table = next;
    }
  }

  template <class Func>
  V upsert(const K &key, const V & value, Func func, const std::size_t thread_id) {
    assert(key != KT::NullKey and key != KT::DeadKey and value != VT::NullValue and value != VT::Tombstone and value != VT::RedirectValue and !VT::is_primed(value));
    epoch_reclaimer::epoch_guard guard(m_reclaimer.get_epoch(thread_id));
    Table *root_table = m_root.load(std::memory_order_consume);
    Table *current_table = root_table;
    assert(current_table != nullptr);
  attemptInsert:
    std::pair<ClaimOperationResult, Bucket *> result = Table::claim_bucket(current_table, key);
    ClaimOperationResult op_result = result.first;
    Bucket * bucket = result.second;
    if(op_result == ClaimOperationResult::Claimed) {
      // Try to update value...
      while(true) {
        V raw_value = bucket->value.load(std::memory_order_relaxed);
        V cur_value = VT::get_value(raw_value);
        // Redirect value
        if(cur_value == VT::RedirectValue) {
          copy_table(current_table);
          current_table = load_next_table(current_table);
          upgrade_table(guard);
          goto attemptInsert;
        }
        // Primed value
        if(VT::is_primed(raw_value)) {
          // Someone is copying the table, go help. Then retry this whole mess.
          copy_table(current_table);
          // Make SURE our bucket is moved over.
          Table * next_table = load_next_table(current_table);
          copy_bucket(current_table, bucket, next_table);
          current_table = next_table;
          // Try update the table pointer.
          upgrade_table(guard);
          goto attemptInsert;
        }
        // Null value
        if(cur_value == VT::NullValue){
          if(bucket->value.compare_exchange_weak(raw_value, value, std::memory_order_relaxed, std::memory_order_relaxed)){
            return cur_value;
          }
          continue;
        }
        // Some value
        V calculated_value = func(cur_value);
        if(cur_value == calculated_value) {
          // Save ourselves some cycles and simply return.
          return value;
        }
        if(bucket->value.compare_exchange_weak(raw_value, calculated_value, std::memory_order_relaxed, std::memory_order_relaxed)) {
          // Successfully swapped values, return old one.
          if(cur_value == VT::Tombstone) {
            current_table->m_tombstoned_buckets.fetch_add(-1);
          }
          return cur_value;
        }
      }
    } else if(op_result == ClaimOperationResult::GrowthNeeded or op_result == ClaimOperationResult::GrowthOngoing) {
      copy_table(current_table);
      // Try update the table pointer.
      upgrade_table(guard);
      Table * next_table = load_next_table(current_table);
      if(bucket != nullptr) {
        // Make SURE our bucket is moved over if it exists.
        copy_bucket(current_table, bucket, next_table);
      }
      current_table = next_table;
      goto attemptInsert;
    }
  }

  V insert(const K &key, const V &value, const std::size_t thread_id) {
    return this->upsert(key, value, [value](const V & cur_value) -> V { return value; }, thread_id);
  }

  
  std::pair<bool, V> remove(const K &key, const std::size_t thread_id) {
    Table * table = m_root.load(std::memory_order_consume);
    assert(table != nullptr);
    epoch_reclaimer::epoch_guard guard(m_reclaimer.get_epoch(thread_id));
    while(true) {
      Bucket * bucket = Table::find(table, key);
      if(bucket != nullptr) {
        V raw_value = bucket->value.load(std::memory_order_relaxed);
        V cur_value = VT::get_value(raw_value);
      removeLoadBegin:
        // Someone who hasn't been copied and has been fully committed into the table.
        if(cur_value != VT::RedirectValue and cur_value != VT::NullValue and cur_value != VT::Tombstone and !VT::is_primed(raw_value)) {
          if(!bucket->value.compare_exchange_weak(raw_value, VT::Tombstone, std::memory_order_relaxed, std::memory_order_relaxed)) {
            goto removeLoadBegin;
          }
          table->m_tombstoned_buckets.fetch_add(1);
          return std::make_pair(true, cur_value);
        }
      }
      Table *next = table->m_next_table.load(std::memory_order_consume);
      if(next == nullptr) { // No other tables.
        V null_value = VT::NullValue;
        return std::make_pair(false, null_value);
      }
      table = next;
    }
  }

  std::size_t size() {
    return m_root.load(std::memory_order_consume)->m_size;
  }
  
  bool verify_table() {
    Table *current = m_root.load(std::memory_order_consume);
    std::size_t table_no = 1;
    std::size_t last_size = current->m_size;
    while(current != nullptr) {
      last_size = current->m_size;
      Table *next = current->m_next_table.load(std::memory_order_consume);
      if(next != nullptr) {
        assert(current->m_size < next->m_size && (current->m_size * 2) == next->m_size);
      }
      for(std::size_t i = 0; i < current->m_size; i++) {
        K cur_key = current->m_buckets[i].key.load();
        V cur_value = current->m_buckets[i].value.load();
        if(cur_key != KT::NullKey and cur_key != KT::DeadKey) {
          if(cur_value == VT::RedirectValue and next == nullptr) {
            std::cout << "REDIRECT VALUE IN THE FINAL TABLE" << std::endl;
            return false; // Redirect in final table.
          } else if(cur_value != VT::RedirectValue and next != nullptr) {
            std::cout << "NON-REDIRECT VALUE IN MIDDLE TABLE: " << cur_key <<  " " << cur_value << " " << current->m_size << " " << table_no << std::endl;
            return false;
          }
        }
        if(cur_key == KT::DeadKey and next == nullptr) {
          std::cout << "DEAD KEY REDIRECT IN FINAL TABLE" << std::endl;
          return false; // Redirect in final table
        }
      }
      std::cout << "Table number: " << table_no  << " copied: " << current->m_buckets_copied.load() << std::endl;
      current = next;
      if(current != nullptr and (last_size * 2) != current->m_size) {
        std::cout << "NON UNIFORM TABLE GROWTH" << std::endl;
      }
      table_no++;
    }
    return true;
  }
};


template <class K, class V, class KT = utils::KeyTraits<K>, class VT = utils::ValueTraits<V>>
using lf_lp = table_manager<lf_linear_probing, K, V, KT, VT>;


} // End of concurrent

}}
