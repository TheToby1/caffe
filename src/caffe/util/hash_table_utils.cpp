#include "caffe/util/hash_table_utils.h"

std::size_t HT::utils::distance_from_slot(std::size_t table_size,
                               std::size_t original_slot,
                               std::size_t current_slot) {
  return (current_slot < original_slot)
             ? table_size - (original_slot - current_slot)
             : current_slot - original_slot;
}