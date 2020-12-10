#ifndef ACTIVATION_CUH_
#define ACTIVATION_CUH_

// Enable CUDA member functions
#define ZHP_CUDA

// Engine headers
#include <activation.hpp>

namespace zhetapi {

	namespace ml {

		template <class T>
		__host__ __device__
		Vector <T> Activation <T> ::operator()(const Vector <T> &x) const
		{
			return x;
		}

		template <class T>
		__host__ __device__
		Activation <T> *Activation <T> ::derivative() const
		{

#ifndef __CUDA_ARCH__

			return new Activation();

#else

			Activation *act;

			cudaMalloc((void **) &act, sizeof(Activation));

			*act = Activation();

			return act;

#endif

		}

	}

}

#endif
