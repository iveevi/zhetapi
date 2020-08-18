namespace ml {

	// ReLU activation
	template <class T>
	T relu(T x)
	{
		return max(x, 0);
	}

	// Leaky ReLU activation
	template <class T>
	T leaky_relu(T x, T alpha)
	{
		if (x >= 0)
			return x;
		
		return alpha * x;
	}

	template <class T>
	T logistic(T x)
	{
		return 1/(1 + exp(-x));
	}

}
