#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// C/C++ headers
#include <cmath>
#include <memory>

// Engine headers
#ifndef ZHP_CUDA

#include <vector.hpp>

#else

#include <cuda/vector.cuh>

#endif

namespace zhetapi {
		
	namespace ml {

		template <class T>
		class Optimizer {
		public:
			
			enum optimizer_type {
				OPT_Default,
				OPT_SE,
				OPT_MSE,
			};

#ifndef ZHP_CUDA
			
			Optimizer();

			virtual Vector <T> operator()(const Vector <T> &, const Vector <T> &) const;

			virtual Optimizer *derivative() const;
			
			int get_optimizer_type() const;

#else

			__host__ __device__
			Optimizer();

			__host__ __device__
			virtual Vector <T> operator()(const Vector <T> &, const Vector <T> &) const;
			
			__host__ __device__
			virtual Optimizer *derivative() const;
		
			__host__ __device__
			int get_optimizer_type() const;

			template <class U>
			__host__ __device__
			friend Optimizer <U> *copy(Optimizer <U> *);

#endif

		protected:
			optimizer_type kind;
		};

#ifndef ZHP_CUDA

		template <class T>
		Optimizer <T> ::Optimizer() : kind(OPT_Default) {}

		template <class T>
		Vector <T> Optimizer <T> ::operator()(const Vector <T> &comp, const Vector <T> &in) const
		{
			return {(comp - in).norm()};
		}

		template <class T>
		Optimizer <T> *Optimizer <T> ::derivative() const
		{
			return new Optimizer();
		}

		template <class T>
		int Optimizer <T> ::get_optimizer_type() const
		{
			return kind;
		}

#endif

	}

}

#endif
