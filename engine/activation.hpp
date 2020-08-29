#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

// Engine headers
#include <algorithm>
#include <functional>

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

		virtual T operator()(const T &) const;

		virtual Activation *derivative() const;
	};

	template <class T>
	Activation <T> ::Activation() {}

	template <class T>
	T Activation <T> ::operator()(const T &x) const
	{
		return T (0);
	}

	template <class T>
	Activation <T> *Activation <T> ::derivative() const
	{
		return new Activation();
	}

}

#endif
