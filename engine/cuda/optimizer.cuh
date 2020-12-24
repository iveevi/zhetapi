#ifndef OPTIMIZER_CUH_
#define OPTIMIZER_CUH_

// Enable CUDA member functions
#define ZHP_CUDA

// Engine headers
#include <optimizer.hpp>

#include <cuda/vector.cuh>

namespace zhetapi {

	namespace ml {
		
		template <class T>
		__host__ __device__
		Optimizer <T> ::Optimizer() {}
		
		template <class T>
		__host__ __device__
		Vector <T> Optimizer <T> ::operator()(const Vector <T> &comp, const Vector <T> &in) const
		{
			return Vector <T> (1, (comp - in).norm());
		}

		template <class T>
		__host__ __device__
		Optimizer <T> *Optimizer <T> ::derivative() const
		{
			return new Optimizer();
		}

		template <class T>
		__host__ __device__
		int Optimizer <T> ::get_optimizer_type() const
		{
			return kind;
		}

	}

}

#endif
