#ifndef TSQUEUE_H_
#define TSQUEUE_H_

namespace zhetapi {

namespace ads {

// Thread safe queue class
template <class T>
class TSQueue {
	size_t		_cap = 0;
	size_t		_last = 0;
	T *		_arr = nullptr;
	std::mutex	_lock;
public:
	// Initialize to eight elements,
	// double the capacity whenever
	// it needs to be increased
	TSQueue() : _cap(8) {
		_arr = (T *) malloc(_cap * sizeof(T));
	}

	~TSQueue() {
		delete[] _arr;
	}

	T pop() {
		if (_last <= 0)
			throw empty_queue();
		
		_lock.lock();
		T ret = _arr[--_last];
		_lock.unlock();

		return ret;
	}

	// Returns true if it had to resize
	bool push(T val) {
		_lock.lock();
		_arr[_last++] = val;

		if (_last >= _cap) {
			_cap <<= 1;
			_arr = (T *) realloc(_arr, _cap * sizeof(T));
			_lock.unlock();

			return true;
		}

		_lock.unlock();
		return true;
	}

	size_t size() const {
		return _last + 1;
	}

	size_t empty() const {
		return (_last == 0);
	}

	/* void print() const {
		cout << "| ";
		for (size_t i = 0; i < _cap; i++) {
			if (i < _last)
				cout << _arr[i];
			else
				cout << "x";
			
			if (i < _cap - 1)
				cout << " | ";
			else
				cout << " |" << endl;
		}
	} */
	
	// Execptions
	class empty_queue : std::runtime_error {
	public:
		empty_queue()
			: std::runtime_error("Cannot pop"
				" from empty TSQueue") {}
	};
};

}

}

#endif