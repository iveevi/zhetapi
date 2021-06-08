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

// A memory pool class
class NVArena {
public:
	using memmap = std::map <void *, size_t, __addr_cmp>;

	// TODO: bad alloc

	// Thrown if the address has never been allocated
	class segfault : public std::runtime_error {
	public:
		segfault() : std::runtime_error("NVArena: segmentation fault.") {}
	};

	// Thrown if the address was allocated but then already freed
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

template <class T>
T *NVArena::alloc(size_t items)
{
        void *data = alloc(items * sizeof(T));

        return (T *) data;
}

template <class T>
void NVArena::free(T *ptr)
{
	free((void *) ptr);
}

// Transfers n items of type T, not n bytes
template <class T>
void NVArena::write(T *dst, T *src, size_t n)
{
	write((void *) dst, (void *) src, n * sizeof(T));
}

// Transfers n items of type T, not n bytes
template <class T>
void NVArena::read(T *dst, T *src, size_t n)
{
	read((void *) dst, (void *) src, n * sizeof(T));
}

}

#endif
