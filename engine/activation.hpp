#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

// C/C++ headers
#include <algorithm>
#include <functional>

// Engine headers
#include <vector.hpp>

namespace zhetapi {

	namespace ml {

		/*
		* Scalar activation
		*
		* @tparam T the input and output type of the activation
		*/
		template <class T>
		class Activation {
		public:
			Activation();

#ifndef ZHP_CUDA

			virtual Vector <T> operator()(const Vector <T> &) const;

			virtual Activation *derivative() const;

#else

			__host__ __device__
			virtual Vector <T> operator()(const Vector <T> &) const;

			__host__ __device__
			virtual Activation *derivative() const;

#endif

		};
		
		template <class T>
		Activation <T> ::Activation() {}

#ifndef ZHP_CUDA

		template <class T>
		Vector <T> Activation <T> ::operator()(const Vector <T> &x) const
		{
			return x;
		}

		template <class T>
		Activation <T> *Activation <T> ::derivative() const
		{
			return new Activation();
		}

#endif

	}

}

#endif
