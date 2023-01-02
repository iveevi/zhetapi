#ifndef ZHETAPI_ALLOCATOR_H_
#define ZHETAPI_ALLOCATOR_H_

// Standard headers
#include <cstddef>
#include <iostream>
#include <memory>
#include <unordered_map>

namespace zhetapi {

namespace detail {

// Memory allocation tracker
// TODO: make thread safe
class MemoryTracker {
	long long int m_allocs = 0;
	long long int m_frees = 0;
	long long int m_inuse = 0;

	std::unordered_map <void *, size_t> m_map;

	// TODO: variant based allocation
	template <class T>
	T *alloc(size_t elements) {
		// TODO: option to throw or not
		if (elements == 0)
			throw std::runtime_error("Must allocate non-zero number of elements");

		T *ptr = new T[elements];
		
		m_allocs++;
		m_inuse += elements * sizeof(T);
		m_map[ptr] = elements * sizeof(T);
		
		return ptr;
	}

	template <class T>
	void free(T *ptr) {
		if (m_map.find(ptr) == m_map.end())
			throw std::runtime_error("Attempt to free unallocated memory");

		m_frees++;
		m_inuse -= m_map[ptr];
		m_map.erase(ptr);

		delete[] ptr;
	}

	static MemoryTracker &one() {
		static MemoryTracker singleton;
		return singleton;
	}
public:
	static void report() {
		MemoryTracker &t = one();

		// TODO: table
		double MB = 1024.0 * 1024.0;
		std::cout << "Memory allocation report:" << std::endl;
		std::cout << "\tAllocations: " << t.m_allocs
			<< ", Frees: " << t.m_frees
			<< ", Net: " << t.m_allocs - t.m_frees << std::endl;
		std::cout << "\tIn use: " << t.m_inuse/MB << " MB" << std::endl;
	}

	template <class T>
	friend T *allocate(size_t);

	template <class T>
	friend void deallocate(T *);
};

template <class T>
T *allocate(size_t n)
{
	return MemoryTracker::one().alloc <T> (n);
}

template <class T>
void deallocate(T *ptr)
{
	MemoryTracker::one().free(ptr);
}

template <class T>
std::shared_ptr <T []> make_shared_array(size_t elements)
{
	return std::shared_ptr <T []> (
		allocate <T> (elements),
		deallocate <T>
	);
}

}

}

#endif
