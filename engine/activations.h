#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

#include <algorithm>

namespace ml {

	// Scalar activation
	template <class T>
	class Activation {
	public:
		// Aliases
		using ftr = T (*)(T);
	private:
		// Activation
		ftr act;

		// Derivative of Activation
		ftr d_act;
	public:
		Activation(ftr, ftr);

		T operator()(T);

		const Activation &derivative() const;

		// Exceptions
		class null_activation {};
	};

	template <class T>
	Activation <T> ::Activation(ftr a, ftr b) : act(a), d_act(b)
	{
		if (a == nullptr)
			throw null_activation();
	}

	template <class T>
	T Activation <T> ::operator()(T x)
	{
		return (*act)(x);
	}

	template <class T>
	const Activation <T> &Activation <T> ::derivative() const
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
	class ReLU : public Activation <T> {
	public:
		ReLU();
	};

	template <class T>
	ReLU <T> ::ReLU() : Activation<T> (&__relu, &__d_relu) {}

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
	T __logistic(T x)
	{
		return 1/(1 + exp(-x));
	}

}

#endif