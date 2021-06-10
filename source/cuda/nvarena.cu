#include <cuda/nvarena.cuh>

#include <iostream>

namespace zhetapi {

/**
 * @brief Initializes the allocator with a specific amount of memory.
 *
 * @param mb the number of megabytes (not bytes!) that the allocator should hold
 * on to and serve.
 */
NVArena::NVArena(size_t mb)
{
	size_t bytes = mb << 20;

	cudaMalloc(&_pool, bytes);

	__cuda_check_error();
}

/**
 * @brief Deconstructor. The allocator releases its pool of memory and notifies
 * the user on blocks of memory that are still allocated.
 */
NVArena::~NVArena()
{
	if (_warn) {
		for (const auto &pr : _flist) {
			if (pr.second != 0) {
				std::cout << "NVArena: untracked block @"
					<< pr.first << " [size=" << pr.second
					<< "]" << std::endl;
			}
		}
	}

	cudaFree(_pool);
}

/**
 * @brief Allocates a block of memory.
 *
 * @param bytes the number of bytes of allocate.
 * 
 * @return the allocated block.
 */
void *NVArena::alloc(size_t bytes)
{
	// Case where __flist is empty
	if (_flist.empty()) {
		// Assign to the free list
		_flist[_pool] = bytes;

		return _pool;
	}

	// Get the last block
	auto last = _flist.rbegin();

	// TODO: throw bad_alloc if there is not more space

	// Allocation strategy: allocate from the end of the arena
	void *laddr = last->first + last->second;

	// Assign to the free list
	_flist[laddr] = bytes;

	return laddr;
}

/**
 * @brief Frees a block of memory.
 *
 * @param ptr the block of memory to be freed.
 */
void NVArena::free(void *ptr)
{
	if (_flist.find(ptr) == _flist.end())
		throw segfault();

	if (_flist[ptr] == 0)
		throw double_free();

	_flist[ptr] = 0;
}

/**
 * @brief Copies a block of memory from host memory to GPU memory, using \c
 * cudaMemcpy. Warns if the number of bytes to copy exceeds the block size on
 * the GPU (assuming the allocators warning flag is turned on).
 *
 * @param dst the pointer to the destination in GPU memory.
 * @param src the pointer to the block in host memory.
 * @param bytes the number of bytes to copy.
 */
void NVArena::write(void *dst, void *src, size_t bytes)
{
	// Do some checks before copying
	if (_warn) {
		auto lb = _flist.lower_bound(dst);

		if (lb == _flist.end()) {
			std::cout << "NVArena: @" << dst
				<< " was never allocated"
				<< std::endl;
		} else {
			void *lim = lb->first + lb->second;

			if (dst + bytes > lim) {
				std::cout << "NVArena: writing " << bytes
					<< " bytes to block @" << lb->first
					<< " [offset +"
					<< ((char *) dst - (char *) lb->first)
					<< "] with only " << lb->second
					<< " bytes allocated" << std::endl;
			}
		}
	}

	cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);

	__cuda_check_error();
}

/**
 * @brief Copies a block of memory from GPU memory to host memory, using \c
 * cudaMemcpy. Warns if the number of bytes to copy exceeds the block size on
 * the GPU (assuming the allocators warning flag is turned on).
 *
 * @param dst the pointer to the destination in host memory.
 * @param src the pointer to the block in GPU memory.
 * @param bytes the number of bytes to copy.
 */
void NVArena::read(void *dst, void *src, size_t bytes)
{
	// Do some checks before copying
	if (_warn) {
		auto lb = _flist.lower_bound(src);

		if (lb == _flist.end()) {
			std::cout << "NVArena: @" << src
				<< " was never allocated"
				<< std::endl;
		} else {
			void *lim = lb->first + lb->second;

			if (src + bytes > lim) {
				std::cout << "NVArena: read " << bytes
					<< " bytes from block @" << lb->first
					<< " [offset +"
					<< ((char *) src - (char *) lb->first)
					<< "] with only " << lb->second
					<< " bytes allocated" << std::endl;
			}
		}
	}

	cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);

	__cuda_check_error();
}

/**
 * @brief Prints each block that has allocated (or freed). Use for debugging
 * purposes.
 */
void NVArena::show_mem_map() const
{
	for (const auto &pr : _flist) {
		std::cout << "block @" << pr.first << ": " << pr.second
			<< " bytes";

		if (pr.second == 0)
			std::cout << "\t[freed]";

		std::cout << std::endl;
	}
}

}
