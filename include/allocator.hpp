#ifndef ZHETAPI_ALLOCATOR_H_
#define ZHETAPI_ALLOCATOR_H_

// Standard headers
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <unordered_map>

// Check CUDA availability
#ifdef __CUDACC__
#define ZHETAPI_CUDA 1
#else
#define ZHETAPI_CUDA 0
#endif

namespace zhetapi {

// Memory variant
enum Variant {
	eCPU,
	eCUDA // TODO: only enable if CUDA is available
};

namespace detail {

// Memory allocation tracker
// TODO: make thread safe
class MemoryTracker {
	// TODO: pack into structs...
	long long int m_cpu_allocs = 0;
	long long int m_cpu_frees = 0;
	long long int m_cpu_inuse = 0;
	
	std::unordered_map <void *, size_t> m_cpu_map;

	long long int m_cuda_allocs = 0;
	long long int m_cuda_frees = 0;
	long long int m_cuda_inuse = 0;

	std::unordered_map <void *, size_t> m_cuda_map;

	// TODO: variant based allocation
	template <class T>
	T *alloc(size_t elements, Variant variant) {
		// TODO: option to throw or not
		if (elements == 0)
			throw std::runtime_error("Must allocate non-zero number of elements");

		T *ptr = nullptr;
		if (variant == eCPU) {
			ptr = new T[elements];
			
			m_cpu_allocs++;
			m_cpu_inuse += elements * sizeof(T);
			m_cpu_map[ptr] = elements * sizeof(T);
		} else if (variant == eCUDA) {
			if constexpr (!ZHETAPI_CUDA)
				throw std::runtime_error("CUDA is not available");

#ifdef __CUDACC__
			cudaMalloc(&ptr, elements * sizeof(T));

			m_cuda_allocs++;
			m_cuda_inuse += elements * sizeof(T);
			m_cuda_map[ptr] = elements * sizeof(T);
#endif
		}
		
		return ptr;
	}

	template <class T>
	void deallocate(T *ptr, Variant variant) {
		if (variant == eCPU) {
			if (m_cpu_map.find(ptr) == m_cpu_map.end())
				throw std::runtime_error("Attempt to free unallocated memory");

			m_cpu_frees++;
			m_cpu_inuse -= m_cpu_map[ptr];
			m_cpu_map.erase(ptr);

			delete[] ptr;
		} else if (variant == eCUDA) {
			if constexpr (!ZHETAPI_CUDA)
				throw std::runtime_error("CUDA is not available");

#ifdef __CUDACC__
			if (m_cuda_map.find(ptr) == m_cuda_map.end())
				throw std::runtime_error("Attempt to free unallocated memory");

			m_cuda_frees++;
			m_cuda_inuse -= m_cuda_map[ptr];
			m_cuda_map.erase(ptr);

			cudaFree(ptr);

#endif
		}
	}

	template <class T>
	void copy(T *dst, T *src, size_t elements, Variant variant) {
		if (variant == eCPU) {
			if (m_cpu_map.find(dst) == m_cpu_map.end())
				throw std::runtime_error("Attempt to copy to unallocated memory");

			if (m_cpu_map.find(src) == m_cpu_map.end())
				throw std::runtime_error("Attempt to copy from unallocated memory");

			std::copy(src, src + elements, dst);
		} else if (variant == eCUDA) {
			if constexpr (!ZHETAPI_CUDA)
				throw std::runtime_error("CUDA is not available");

#ifdef __CUDACC__
			if (m_cuda_map.find(dst) == m_cuda_map.end())
				throw std::runtime_error("Attempt to copy to unallocated memory");

			if (m_cuda_map.find(src) == m_cuda_map.end())
				throw std::runtime_error("Attempt to copy from unallocated memory");

			cudaMemcpy(dst, src, elements * sizeof(T), cudaMemcpyDeviceToDevice);
#endif
		}
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

		std::cout << "\tAllocations: " << t.m_cpu_allocs
			<< ", Frees: " << t.m_cpu_frees
			<< ", Net: " << t.m_cpu_allocs - t.m_cpu_frees << std::endl;
		std::cout << "\tIn use: " << t.m_cpu_inuse/MB << " MB" << std::endl;

		if constexpr (ZHETAPI_CUDA) {
			std::cout << "\n\tCUDA Allocations: " << t.m_cuda_allocs
				<< ", CUDA Frees: " << t.m_cuda_frees
				<< ", Net: " << t.m_cuda_allocs - t.m_cuda_frees << std::endl;
			std::cout << "\tCUDA In use: " << t.m_cuda_inuse/MB << " MB" << std::endl;
		}
	}

	template <class T>
	friend T *allocate(size_t, Variant);

	template <class T>
	friend void deallocate(T *, Variant);

	template <class T>
	friend void copy(const std::shared_ptr <T []> &,
			const std::shared_ptr <T []> &,
			size_t, Variant);
};

template <class T>
T *allocate(size_t n, Variant variant)
{
	return MemoryTracker::one().alloc <T> (n, variant);
}

template <class T>
void deallocate(T *ptr, Variant variant)
{
	MemoryTracker::one().deallocate(ptr, variant);
}

template <class T>
std::shared_ptr <T []> make_shared_array(size_t elements, Variant variant)
{
	return std::shared_ptr <T []> (
		allocate <T> (elements, variant),
		[variant](T *ptr) {
			deallocate(ptr, variant);
		}
	);
}

template <class T>
void copy(const std::shared_ptr <T []> &dst,
		const std::shared_ptr <T []> &src,
		size_t elements, Variant variant)
{
	MemoryTracker::one().copy(dst.get(), src.get(), elements, variant);
}

}

}

#endif
