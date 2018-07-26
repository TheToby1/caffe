#ifndef HASH_TABLE_UTILS_H_
#define HASH_TABLE_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <functional>

namespace HT {

namespace utils {

template <typename T> T nearest_power_of_two(T num) {
  std::size_t actual_size = sizeof(num);
  if (num == 0)
    return std::size_t(0);
  num--;
  // Single byte switches
  num |= num >> 1;
  num |= num >> 2;
  num |= num >> 4;
  std::size_t current_shift = 8;
  for (std::size_t i = 1; i < actual_size; i *= 2, current_shift *= 2) {
    num |= num >> current_shift;
  }
  return ++num;
}

template <class T> struct KeyTraits {
  typedef T Key;
  typedef typename std::hash<T> Hash;
  static const Key NullKey = std::numeric_limits<Key>::max();
  static const Key DeadKey = std::numeric_limits<Key>::max() - 1;
  static std::size_t hash(T key) { return Hash{}(key); }
};

template <> struct KeyTraits<std::uint64_t> {
  typedef std::uint64_t Key;
  static const Key NullKey = std::numeric_limits<Key>::max();
  static const Key DeadKey = std::numeric_limits<Key>::max() - 1;
  static std::size_t hash(Key key) {
    key = (key ^ (key >> 30)) * std::uint64_t(0xbf58476d1ce4e5b9);
    key = (key ^ (key >> 27)) * std::uint64_t(0x94d049bb133111eb);
    key = key ^ (key >> 31);
    return key;
  }
};

template <class T> struct ValueTraits {
  typedef T Value;
  static const Value PrimedBit = Value(1) << ((sizeof(Value) * 8) - 1); // "I'm being relocated" bit-mask.
  static const Value NullValue = std::numeric_limits<Value>::max() & (~PrimedBit);
  static const Value Tombstone = NullValue - 1;
  static const Value RedirectValue = NullValue - 2; // ABANDON SHIP!
  static bool is_primed(Value value) {
    return (value & PrimedBit) == PrimedBit;
  }
  static Value get_value(Value value) {
    return value & (~PrimedBit);
  }
  static Value make_primed(Value value) {
    return value | PrimedBit;
  }
};

std::size_t distance_from_slot(std::size_t table_size,
                               std::size_t original_slot,
                               std::size_t current_slot);

}}

#endif