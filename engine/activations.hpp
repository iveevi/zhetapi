#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

// Engine headers
#include <algorithm>
#include <functional>

namespace ml {

	// Scalar activation
	template <class T, class U>
	class Activation {
	public:
		// Aliases
		using ftr = T;
	protected:
		// Activation
		ftr act;

		// Derivative of Activation
		ftr d_act;
	public:
		Activation(ftr, ftr);

		T operator()(U);

		const Activation &derivative() const;

		// Exceptions
		class null_activation {};
	};

	template <class T, class U>
	Activation <T, U> ::Activation(ftr a, ftr b) : act(a), d_act(b)
	{
		if (a == nullptr)
			throw null_activation();
	}

	template <class T, class U>
	T Activation <T, U> ::operator()(U x)
	{
		return act(x);
	}

	template <class T, class U>
	const Activation <T, U> &Activation <T, U> ::derivative() const
	{
		return Activation(d_act, nullptr);
	}

	// ReLU activation functions
	template <class T>
	T __relu(T x)
	{
		return std::max(x, (T) 0);
	}

	template <class T>
	T __d_relu(T x)
	{
		if (x > 0)
			return 1;
		
		return 0;
	}

	// ReLU activation class
	template <class T>
	using __unary = std::function <T (T)>;

	template <class T>
	class ReLU : public Activation <__unary <T>, T> {
	public:
		ReLU();
	};

	template <class T>
	ReLU <T> ::ReLU() : Activation <__unary <T>, T> (&__relu, &__d_relu) {}

	// Leaky ReLU activation function
	template <class T>
	T __leaky_relu(T x, T alpha)
	{
		if (x >= 0)
			return x;
		
		return alpha * x;
	}

	template <class T>
	T __d_leaky_relu(T x, T alpha)
	{
		if (x > 0)
			return 1;
		
		return alpha;
	}

	/*

	// Derivate of Leaky ReLU as a class
	template <class T>
	class __DLeakyReLU : public Activation <T> {
		T alpha;
	public:
		__DLeakyReLU(T);

		T operator()(T);
	};

	template <class T>
	__DLeakyReLU <T> ::__DLeakyReLU(T al) : alpha(al),
			Activation <T> (&__d_leaky_relu, nullptr) {}

	template <class T>
	T __DLeakyReLU <T> ::operator()(T x)
	{
		return this->act(x, alpha);
	}

	// Leaky ReLU class
	template <class T>
	class LeakyReLU : public Activation <T> {
		T alpha;
	public:
		LeakyReLU(T);

		T operator()(T);
		
		const Activation <T> &derivative() const;
	};

	template <class T>
	LeakyReLU <T> ::LeakyReLU(T al) : alpha(al),
			Activation<T> (&__leaky_relu, &__d_leaky_relu) {}
	
	template <class T>
	T LeakyReLU <T> ::operator()(T x)
	{
		return this->act(x, alpha);
	}

	template <class T>
	const Activation <T> &LeakyReLU <T> ::derivative() const
	{
		return __DLeakyReLU <T> (alpha);
	}

	// Logitstic activation function
	template <class T>
	T __sigmoid(T x)
	{
		return 1/(1 + exp(-x));
	}

	template <class T>
	T __d_sigmoid(T x)
	{
		T tmp = 1/(1 + exp(-x));

		return tmp * (T (1) - tmp);
	}

	// Sigmoid class
	template <class T>
	class Sigmoid : public Activation <T> {
	public:
		Sigmoid();
	};

	template <class T>
	Sigmoid <T> ::Sigmoid() : Activation <T> (&__sigmoid, &__d_sigmoid) {}

	// Logitstic activation function with scaling
	template <class T>
	T __scaled_sigmoid(T x, T alpha)
	{
		return 1/(1 + exp(-alpha * x));
	}

	template <class T>
	T __d_scaled_sigmoid(T x, T alpha)
	{
		return alpha/(2 * cosh(alpha * x) + 2);
	}

	// Derivate of Scaled Sigmoid as a class
	template <class T>
	class __DScaledSigmoid : public Activation <T> {
		T alpha;
	public:
		__DScaledSigmoid(T);

		T operator()(T);
	};

	template <class T>
	__DScaledSigmoid <T> ::__DScaledSigmoid(T al) : alpha(al),
			Activation <T> (&__d_scaled_sigmoid, nullptr) {}

	template <class T>
	T __DScaledSigmoid <T> ::operator()(T x)
	{
		return this->act(x, alpha);
	}

	// Scaled Sigmoid class
	template <class T>
	class ScaledSigmoid : public Activation <T> {
		T alpha;
	public:
		ScaledSigmoid(T);

		T operator()(T);
		
		const Activation <T> &derivative() const;
	};

	template <class T>
	ScaledSigmoid <T> ::ScaledSigmoid(T al) : alpha(al),
			Activation<T> (&__scaled_sigmoid, &__d_scaled_sigmoid) {}
	
	template <class T>
	T ScaledSigmoid <T> ::operator()(T x)
	{
		return this->act(x, alpha);
	}

	template <class T>
	const Activation <T> &ScaledSigmoid <T> ::derivative() const
	{
		return __DScaledSigmoid <T> (alpha);
	}

	*/
}

#endif