#ifndef STD_ACTIVATION_FUNCTIONS_H_
#define STD_ACTIVATION_FUNCTIONS_H_

namespace ml {

	// Linear activation functions
	template <class T>
	T __linear(T x, T alpha)
	{
		return alpha * x;
	}

	template <class T>
	T __d_linear(T x, T alpha)
	{
		return alpha;
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

	// Leaky ReLU activation functions
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

	// Sigmoid activation function
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

	// Sigmoid activation function with scaling
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

}

#endif
