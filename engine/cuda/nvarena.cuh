#ifndef NVARENA_CUH_
#define NVARENA_CUH_

#define ZHP_CUDA

// Engine headers
#include <cuda/essentials.cuh>

namespace zhetapi {

// A memory pool class
class NVArena {
	void	*__pool;
public:
	NVArena(size_t);

	~NVArena();
};

}

#endif
