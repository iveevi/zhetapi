#ifndef ZHETAPI_ALLOCATOR_H_
#define ZHETAPI_ALLOCATOR_H_

// Standard headers
#include <cstddef>
#include <unordered_map>
#include <iostream>

namespace zhetapi {

namespace detail {

// Memory allocation tracker
// TODO: make thread safe
class Tracker {
	int m_allocs = 0;
	int m_frees = 0;
	int m_inuse = 0;

	std::unordered_map <void *, size_t> m_map;

	// TODO: variant based allocation
	template <class T>
	T *alloc(size_t elements) {
		T *ptr = new T[elements];
		m_map[ptr] = elements;
		m_allocs++;
		m_inuse += elements;
		return ptr;
	}

	template <class T>
	void free(T *ptr) {
		if (m_map.find(ptr) == m_map.end()) {
			// Log warning...
			std::cerr << "Warning: Attempt to free unallocated memory" << std::endl;
			return;
		}

		m_frees++;
		m_inuse -= m_map[ptr];
		m_map.erase(ptr);
		delete[] ptr;
	}

	static Tracker &one() {
		static Tracker singleton;
		return singleton;
	}

	template <class T>
	friend T *allocate(size_t);

	template <class T>
	friend void deallocate(T *);
};

template <class T>
T *allocate(size_t n)
{
	return Tracker::one().alloc <T> (n);
}

template <class T>
void deallocate(T *ptr)
{
	Tracker::one().free(ptr);
}

}

}

#endif
