#ifndef TSQUEUE_H_
#define TSQUEUE_H_

// Standard headers
#include <queue>

namespace zhetapi {

namespace ads {

// Thread safe queue class,
// wrapper to std::queue <T>
template <class T>
class TSQueue {
	std::queue <T>	_queue;
	std::mutex	_lock;
public:
	TSQueue() {}

	T pop() {
		_lock.lock();
		T ret = _queue.front();
		_queue.pop();
		_lock.unlock();

		return ret;
	}

	// Returns true if it had to resize
	void push(T val) {
		_lock.lock();
		_queue.push(val);
		_lock.unlock();
	}

	size_t size() const {
		return _queue.size();
	}

	// TODO: this is a critical section
	size_t empty() const {
		return _queue.empty();
	}
};

}

}

#endif