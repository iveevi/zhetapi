#ifndef NVARENA_CUH_
#define NVARENA_CUH_

#define ZHP_CUDA

// C/C++ headers
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
	// Whole pool
	void *					__pool = nullptr;

	// Free list (ordered by address)
	std::map <void *, size_t, __addr_cmp>	__flist;
public:
	explicit NVArena(size_t);

	~NVArena();

	void *alloc(size_t);

	// Add casted alloc
	template <class T>
	T *alloc(size_t);

	void memcpy(void *, size_t);
};

template <class T>
T *NVArena::alloc(size_t items)
{
        void *data = alloc(items * sizeof(T));

        return (T *) data;
}

}

#endif
