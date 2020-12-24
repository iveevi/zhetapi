#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

// C/C++ headers
#include <algorithm>
#include <functional>

// Engine headers
#ifndef ZHP_CUDA

#include <vector.hpp>

#else

#include <cuda/vector.cuh>

#endif

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

			enum activation_type {
				AT_Default,
				AT_Linear,
				AT_ReLU,
				AT_Sigmoid
			};

#ifndef ZHP_CUDA
			
			Activation();

			virtual Vector <T> operator()(const Vector <T> &) const;

			virtual Activation *derivative() const;

			int get_activation_type() const;

#else

			__host__ __device__
			Activation();

			__host__ __device__
			virtual Vector <T> operator()(const Vector <T> &) const;

			__host__ __device__
			virtual Activation *derivative() const;

			__host__ __device__
			int get_activation_type() const;

			template <class U>
			__host__ __device__
			friend Activation <U> *copy(Activation <U> *);

#endif

		protected:
			activation_type kind;
		};

#ifndef ZHP_CUDA
		
		template <class T>
		Activation <T> ::Activation() : kind(AT_Default) {}

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

		template <class T>
		int Activation <T> ::get_activation_type() const
		{
			return kind;
		}

#endif

	}

}

#endif
