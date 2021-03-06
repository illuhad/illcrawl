/*
 * This file is part of illcrawl, a reconstruction engine for data from
 * the illustris simulation.
 *
 * Copyright (C) 2017  Aksel Alpay
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef ASYNC_WORKER_HPP
#define ASYNC_WORKER_HPP

#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>

namespace async {

/// A worker thread that executes exactly one task in the background.
/// If a second task is enqueued, waits until the first task
/// has completed.
class worker_thread
{
public:
  /// Construct object
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

  /// If a task is currently running, waits until it
  /// has completed.
  void wait()
  {
    if(_is_operation_pending)
    {
      std::unique_lock<std::mutex> lock(_mutex);
      // Wait until no operation is pending
      _condition_wait.wait(lock, [this]{return !_is_operation_pending;});
    }
  }


  /// Enqueues a user-specified function for asynchronous
  /// execution in the worker thread. If another task is
  /// still pending, waits until this task has completed.
  /// \tparam Function A callable object with signature void(void).
  /// \param f The function to enqueue for execution
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

  /// \return whether there is currently an operation
  /// pending.
  inline
  bool is_currently_working() const
  {
    return _is_operation_pending;
  }
private:

  /// Stop the worker thread - this should only be
  /// done in the destructor.
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

  /// Starts the worker thread, which will execute the supplied
  /// tasks. If no tasks are available, waits until a new task is
  /// supplied.
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
