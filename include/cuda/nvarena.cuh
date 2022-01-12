#ifndef NVARENA_CUH_
#define NVARENA_CUH_

#define ZHP_CUDA

// C/C++ headers
#include <exception>
#include <map>

// Engine headers
#include <cuda/essentials.cuh>

// Namespace external functions
struct __addr_cmp {
	bool operator()(void *a, void *b) {
		return (intptr_t) a < (intptr_t) b;
	}
};

namespace zhetapi {

/**
 * @brief An allocator class for Nvidia GPUs. Has additional features like
 * warnings for memory leaks and copy bound errors. Overall is a more convenient
 * interface to GPU memory than standard CUDA operations like \c cudaMalloc and
 * \c cudaMemcpy.
 */
class NVArena {
public:
	// TODO: need to select the specific GPU
	using memmap = std::map <void *, size_t, __addr_cmp>;

	// TODO: bad alloc

	/**
	 * @brief This exception is thrown if the user tries to free a piece of
	 * memory that was never allocated.
	 */
	class segfault : public std::runtime_error {
	public:
		segfault() : std::runtime_error("NVArena: segmentation fault.") {}
	};

	/**
	 * @brief This exceptoin is thrown if the user frees a piece of memory
	 * more than once. The allocator keeps track of all allocated blocks for
	 * this.
	 */
	class double_free : public std::runtime_error {
	public:
		double_free() : std::runtime_error("NVArena: double free.") {}
	};
private:
	// Whole pool
	void *	_pool	= nullptr;

	// Free list (ordered by address)
	memmap	_flist;

	// Warning flag
	bool _warn	= true;
public:
	explicit NVArena(size_t);

	// Disable copying of any sort
	NVArena(const NVArena &) = delete;
	NVArena &operator=(const NVArena &) = delete;

	~NVArena();

	// Allocation
	void *alloc(size_t = 1);

	template <class T>
	T *alloc(size_t = 1);

	// Deallocation
	void free(void *);

	template <class T>
	void free(T *);

	// TODO: Warn with memcpy
	void write(void *, void *, size_t);
	void read(void *, void *, size_t);

	// void memcpy(void *, size_t);

	// Only allow template for homogenous pointers
	// (no implicit size for heterogenous types)
	template <class T>
	void write(T *, T *, size_t = 1);

	template <class T>
	void read(T *, T *, size_t = 1);

	// Memory map
	void show_mem_map() const;
};

/**
 * @brief Allocates a block of items of a specific type.
 *
 * @tparam t the specific type of item to allocate.
 *
 * @param items the number of items to allocate.
 * 
 * @return the allocated block.
 */
template <class T>
T *NVArena::alloc(size_t items)
{
        void *data = alloc(items * sizeof(T));

        return (T *) data;
}

/**
 * @brief Frees a block of items of a specific type.
 *
 * @tparam T the specific type of item to free.
 *
 * @param ptr the block of memory to be freed.
 */
template <class T>
void NVArena::free(T *ptr)
{
	free((void *) ptr);
}

/**
 * @brief Copies a block of memory from host memory to GPU memory, using \c
 * cudaMemcpy.
 *
 * @tparam T the type of each element in the blocks of memory.
 *
 * @param dst the pointer to the destination in GPU memory.
 * @param src the pointer to the block in host memory.
 * @param n the number of items to copy (note that this copies `n *
 * sizeof(T)` bytes in total).
 */
template <class T>
void NVArena::write(T *dst, T *src, size_t n)
{
	write((void *) dst, (void *) src, n * sizeof(T));
}

/**
 * @brief Copies a block of memory from GPU memory to host memory, using \c
 * cudaMemcpy.
 *
 * @tparam T the type of each element in the blocks of memory.
 *
 * @param dst the pointer to the destination in host memory.
 * @param src the pointer to the block in GPU memory.
 * @param n the number of items to copy (note that this copies `n *
 * sizeof(T)` bytes in total).
 */
template <class T>
void NVArena::read(T *dst, T *src, size_t n)
{
	read((void *) dst, (void *) src, n * sizeof(T));
}

}

#endif
