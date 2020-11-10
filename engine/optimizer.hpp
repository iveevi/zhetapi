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

			virtual Vector <T> operator()(const Vector <T> &, const Vector <T> &) const;

			virtual Optimizer *derivative() const;
		};

		template <class T>
		Optimizer <T> ::Optimizer() {}

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

	}

}

#endif
