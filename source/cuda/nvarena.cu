#include <cuda/nvarena.cuh>

#include <iostream>

namespace zhetapi {

// Allocate per megabyte
NVArena::NVArena(size_t mb)
{
	size_t bytes = mb << 20;

	cudaMalloc(&_pool, bytes);

	__cuda_check_error();
}

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

void NVArena::free(void *ptr)
{
	if (_flist.find(ptr) == _flist.end())
		throw segfault();

	if (_flist[ptr] == 0)
		throw double_free();

	_flist[ptr] = 0;
}

// dst is GPU address, src is host address
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

// dst is host address, src is GPU address
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
