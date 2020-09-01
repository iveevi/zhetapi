#ifndef STD_ACTIVATION_CLASSES_H_
#define STD_ACTIVATION_CLASSES_H_

// Engine headers
#include <activation.hpp>

// Engine std module headers
#include <std_activation_functions.hpp>

namespace ml {

	/*
	 * All activation classes have inlined member functions as the
	 * operations they perform are very minimal. These activation classes
	 * serve only as a wrapper for their underlying functions as well as a
	 * means of extracting their derivatives.
	 *
	 * The obscure naming of the derivative classes is done to encourage
	 * users to use the derivative member function.
	 */
	
	// Linear activation class
	template <class T>
	class __DLinear : public Activation <T> {
		T	__alpha;
	public:
		__DLinear(const T &alpha = T(1)) : __alpha(alpha) {}

		T operator()(const T &x) const {
			return __d_leaky_relu(x, __alpha);
		}
	};
	
	template <class T>
	class Linear : public Activation <T> {
		T	__alpha;
	public:
		Linear(const T &alpha = T(1)) : __alpha(alpha) {}

		T operator()(const T &x) const {
			return __leaky_relu(x, __alpha);
		}

		Activation <T> *derivative() const {
			return new __DLinear <T> (__alpha);
		}
	};

	// ReLU activation class
	template <class T>
	class __DReLU : public Activation <T> {
	public:
		T operator()(const T &x) const {
			return __d_relu(x);
		}
	};

	template <class T>
	class ReLU : public Activation <T> {
	public:
		T operator()(const T &x) const {
			return __relu(x);
		}

		Activation <T> *derivative() const {
			return new __DReLU <T> ();
		}
	};

	// Leaky ReLU activation class
	template <class T>
	class __DLeakyReLU : public Activation <T> {
		T	__alpha;
	public:
		__DLeakyReLU(const T &alpha) : __alpha(alpha) {}

		T operator()(const T &x) const {
			return __d_leaky_relu(x, __alpha);
		}
	};
	
	template <class T>
	class LeakyReLU : public Activation <T> {
		T	__alpha;
	public:
		LeakyReLU(const T &alpha) : __alpha(alpha) {}

		T operator()(const T &x) const {
			return __leaky_relu(x, __alpha);
		}

		Activation <T> *derivative() const {
			return new __DLeakyReLU <T> (__alpha);
		}
	};

	// Sigmoid activation class
	template <class T>
	class __DSigmoid : public Activation <T> {
	public:
		T operator()(const T &x) const {
			return __d_sigmoid(x);
		}
	};

	template <class T>
	class Sigmoid : public Activation <T> {
	public:
		T operator()(const T &x) const {
			return __sigmoid(x);
		}

		Activation <T> *derivative() const {
			return new __DSigmoid <T> ();
		}
	};

	// Scaled Sigmoid activation class
	template <class T>
	class __DScaledSigmoid : public Activation <T> {
		T	__alpha;
	public:
		__DScaledSigmoid(const T &alpha) : __alpha(alpha) {}

		T operator()(const T &x) const {
			return __d_scaled_sigmoid(x, __alpha);
		}
	};
	
	template <class T>
	class ScaledSigmoid : public Activation <T> {
		T	__alpha;
	public:
		ScaledSigmoid(const T &alpha) : __alpha(alpha) {}

		T operator()(const T &x) const {
			return __scaled_sigmoid(x, __alpha);
		}

		Activation <T> *derivative() const {
			return new __DScaledSigmoid <T> (__alpha);
		}
	};

};

#endif
