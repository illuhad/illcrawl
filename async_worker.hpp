#ifndef ASYNC_WORKER_HPP
#define ASYNC_WORKER_HPP

#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>

namespace async {

class worker_thread
{
public:
  worker_thread()
    : _is_operation_pending{false},
      _continue{true},
      _async_operation{[](){}}
  {
    _worker_thread = std::thread{[this](){ work(); } };
  }

  worker_thread(const worker_thread&) = delete;
  worker_thread& operator=(const worker_thread&) = delete;

  ~worker_thread()
  {
    halt();

    if(_worker_thread.joinable())
      _worker_thread.join();
  }


  void wait()
  {
    if(_is_operation_pending)
    {
      std::unique_lock<std::mutex> lock(_mutex);
      // Wait until no operation is pending
      _condition_wait.wait(lock, [this]{return !_is_operation_pending;});
    }
  }


  template<class Function>
  void operator()(Function f)
  {
    wait();

    std::unique_lock<std::mutex> lock(_mutex);
    _async_operation = f;
    _is_operation_pending = true;

    lock.unlock();
    _condition_wait.notify_one();
  }

  inline
  bool is_currently_working() const
  {
    return _is_operation_pending;
  }
private:
  void halt()
  {

    // Wait until no operation is pending
    if(_is_operation_pending)
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _condition_wait.wait(lock, [this]{return !_is_operation_pending;});
    }

    _continue = false;
    _condition_wait.notify_one();
  }

  void work()
  {
    while(_continue)
    {
      {
        std::unique_lock<std::mutex> lock(_mutex);

        // Wait until we have work, or until _continue becomes false
        _condition_wait.wait(lock,
                             [this](){return _is_operation_pending || !_continue;});
      }

      if(_continue && _is_operation_pending)
      {
        _async_operation();

        {
          std::lock_guard<std::mutex> lock(_mutex);
          _is_operation_pending = false;
        }
        _condition_wait.notify_one();
      }
    }
  }

  bool _is_operation_pending;
  std::thread _worker_thread;

  bool _continue;

  std::condition_variable _condition_wait;
  std::mutex _mutex;

  std::function<void()> _async_operation;
};

}

#endif
