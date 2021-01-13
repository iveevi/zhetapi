#ifndef STD_OPTIMIZERS_H_
#define STD_OPTIMIZERS_H_

// Engine headers
#include <optimizer.hpp>

namespace zhetapi {

namespace ml {

template <class T>
class SGD : public Optimizer <T> {
	T	__alpha;
public:
	SGD(T alpha) : __alpha(alpha) {}

	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];

    		Matrix <T> *J = simple_gradient(layers, size, a, z, in, out, cost);
		
		delete[] a;
		delete[] z;

		return J;
	}
};

template <class T>
class Momentum : public Optimizer <T> {
	T		__alpha = 0;
	T		__beta = 0;

	Matrix <T> *	__M = nullptr;
	size_t		__size = 0;
public:
	Momentum(T alpha, T beta) : __alpha(alpha), __beta(beta) {}

	~Momentum() {
		delete[] __M;
	}

	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];

    		Matrix <T> *J = simple_gradient(layers, size, a, z, in, out, cost);
		
		if (size != __size) {
			delete[] __M;

			__size = size;
			__M = new Matrix <T> [__size];
		}
		
		for (size_t i = 0; i < __size; i++) {
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (__M[i].get_dimensions() != odim)
				__M[i] = Matrix <T> (odim.first, odim.second, T(0));
			
			__M[i] = __beta * __M[i] + __alpha * J[i];
		}
		
		delete[] a;
		delete[] z;

		return __M;
	}

	void reset() {
		delete[] __M;

		__size = 0;
		__M = nullptr;
	}
};

template <class T>
class Nesterov : public Optimizer <T> {
	T		__alpha = 0;
	T		__beta = 0;

	Matrix <T> *	__M = nullptr;
	size_t		__size = 0;
public:
	Nesterov(T alpha, T beta) : __alpha(alpha), __beta(beta) {}

	~Nesterov() {
		delete[] __M;
	}

	Matrix <T> *gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
	{
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];

		Layer <T> *adj = new Layer <T> [size];
		
		// Ensure that __M is of the right size
		if (size != __size) {
			delete[] __M;

			__size = size;
			__M = new Matrix <T> [__size];
		}

		for (size_t i = 0; i < size; i++) {
			size_t rs = layers[i].get_fan_out();
			size_t cs = layers[i].get_fan_in() + 1;

			if (__M[i].get_dimensions() != std::make_pair(rs, cs))
				__M[i] = Matrix <T> (rs, cs, T(0));

			adj[i] = layers[i] - __M[i];
		}

    		Matrix <T> *J = simple_gradient(adj, size, a, z, in, out, cost);
		
		for (size_t i = 0; i < __size; i++)
			__M[i] = __beta * __M[i] + __alpha * J[i];
		
		delete[] a;
		delete[] z;
		delete[] adj;

		return __M;
	}

	void reset() {
		delete[] __M;

		__size = 0;
		__M = nullptr;
	}
};

}

}

#endif
