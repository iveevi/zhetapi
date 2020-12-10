#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// C/C++ headers
#include <cmath>
#include <memory>

// Engine headers
#include <vector.hpp>

namespace zhetapi {
		
	namespace ml {

		template <class T>
		class Optimizer {
		public:
			Optimizer();

#ifndef ZHP_CUDA

			virtual Vector <T> operator()(const Vector <T> &, const Vector <T> &) const;

			virtual Optimizer *derivative() const;

#else

			__host__ __device__
			virtual Vector <T> operator()(const Vector <T> &, const Vector <T> &) const;
			
			__host__ __device__
			virtual Optimizer *derivative() const;

#endif

		};

		template <class T>
		Optimizer <T> ::Optimizer() {}

#ifndef ZHP_CUDA

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

#endif

	}

}

#endif
