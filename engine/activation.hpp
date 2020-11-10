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

			virtual Vector <T> operator()(const Vector <T> &) const;

			virtual Activation *derivative() const;
		};

		template <class T>
		Activation <T> ::Activation() {}

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

	}

}

#endif
