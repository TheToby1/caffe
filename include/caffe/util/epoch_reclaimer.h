#ifndef EPOCH_RECLAIMER_H_
#define EPOCH_RECLAIMER_H_ 

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace HT{

namespace CPU{

class epoch_reclaimer {
private:
  static const std::size_t s_num_epochs = 3;
public:
  class epoch {
  private:
    epoch_reclaimer * m_epoch_manager;
    std::atomic_bool m_active;
    std::atomic_size_t m_thread_epoch;
    std::vector<std::function<void()>> m_garbage_list[s_num_epochs];

    void clear_garbage(){
      for (const auto &callback : m_garbage_list[(m_thread_epoch - 2) % s_num_epochs]) {
        callback();
      }CAFFE_UTIL_PERMUTOHEDRAL_H_
      m_garbage_list[(m_thread_epoch - 2) % s_num_epochs].clear();
    }
  public:
    epoch():
      m_epoch_manager(nullptr),
      m_active(false),
      m_thread_epoch(std::size_t(2)){}
    
    void add_garbage(const std::function<void()> &callback) {
      m_garbage_list[(m_thread_epoch - 2) % s_num_epochs].push_back(callback);
    }

    void enter() {
      // this->m_active.store(true); // Active now
      if (this->m_thread_epoch < m_epoch_manager->get_current_epoch()) {
        this->m_thread_epoch++;
        this->clear_garbage();
      }
    }
    void exit() {
      // this->m_active.store(false);
      if (m_epoch_manager->try_incrememnt_current_epoch()) {
        //this->clear_garbage();
      }
    }
    friend class epoch_reclaimer;
  };

  std::atomic_size_t m_global_epoch;
  std::vector<epoch> m_epochs;

public:
  class epoch_guard {
    epoch * m_epoch;
  public:
    epoch_guard(epoch * thread_epoch):
      m_epoch(thread_epoch) { m_epoch->enter(); }
    ~epoch_guard(){ m_epoch->exit(); }
    void add_garbage(const std::function<void()> &callback) {
      m_epoch->add_garbage(callback);
    }
  };

  epoch_reclaimer(std::size_t num_threads = std::thread::hardware_concurrency()):
    m_global_epoch(std::size_t(2)),
    m_epochs(num_threads)
  {
    for(std::size_t i = 0; i < num_threads; i++){
      m_epochs[i].m_epoch_manager = this;
      m_epochs[i].m_thread_epoch = 2;
    }
  }
  std::size_t get_current_epoch() { return m_global_epoch.load(); }
  bool try_incrememnt_current_epoch(){
    std::size_t global_epoch = m_global_epoch.load();
    for(std::size_t i = 0; i < m_epochs.size(); i++) {
      std::size_t current_thread_epoch = m_epochs[i].m_thread_epoch.load();
      if(current_thread_epoch != global_epoch) {
        return false;
      }
    }
    m_global_epoch.compare_exchange_strong(global_epoch, global_epoch + 1);
    return true;
  }
  epoch* get_epoch(const std::size_t thread_id){
    return &m_epochs[thread_id];
  }

};

}}

#endif