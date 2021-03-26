#ifndef MEMORY_H_
#define MEMORY_H_

// Include appropriate headers
#include <mutex>

#include <cuda/essentials.cuh>

#ifdef __CUDACC__

#include <cuda/lock.cuh>

#else

struct Lock {};

#endif

struct _device_property {
	size_t	_nvgpu;
};

enum _mtx_id {
	MTX_CPU,
	MTX_GPU,
	MTX_AVR
};

// Add a source file as well
struct _mutex {
	union {
		// Macro sub mutex
		std::mutex	_cpu_mtx;

		// Make this supportive in gpu and cpu
		// Lock		_gpu_mtx;
	} _mtx;

	_mtx_id _mid;

	_mutex(_mtx_id id) : _mid(id) {
		if (id == MTX_CPU)
			_mtx = std::mutex();
		else if (id == MTX_GPU); // Add gpu id
	}

	void lock() {
#ifdef __CUDA_ARCH__
		// if (id == MTX_GPU)
		//	_mtx._gpu_mtx.lock();
#else
		if (id == MTX_CPU)
			_mtx._cpu_mtx.lock();
#endif
	}

	void unlock() {
#ifdef __CUDA_ARCH__
		// if (id == MTX_GPU)
		//	_mtx._gpu_mtx.lock();
#else
		if (id == MTX_CPU)
			_mtx._cpu_mtx.unlock();
#endif
	}
};

// Work on this later
// Allocater allocates on both gpu and cpu
template <class T>
class Allocator {
protected:
	_device_property *	_dp;
public:
	Allocator() {
		_dp = nullptr;
	}

	T *alloc(size_t size) {
		return new T[size];
	}

	void free(T *ptr) {
		delete[] ptr;
	}

	_device_property *dev_prop() const {
		return _dp;
	}
};

template <class T>
struct _shared_block {
	T *		_addr		= nullptr;
	size_t		_size		= 0;
	size_t *	_refs		= nullptr;
	Allocator <T> *	_allocer	= nullptr;
	_mutex *	_locker		= nullptr;

	_shared_block(size_t size) : _size(size) {
		_addr = _allocer->alloc(size);
		_refs = new size_t;
		
		// CPU only for now
		_locker = new _mutex(MTX_CPU);

		(*_refs)++;
	}
};

#endif