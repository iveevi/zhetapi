#ifndef ACTIVATION_DERIVATIVES_H_
#define ACTIVATION_DERIVATIVES_H_

namespace zhetapi {

namespace ml {

template <class T>
class __DLinear : public Activation <T> {
	T	__alpha;
public:

#ifndef ZHP_CUDA

	__DLinear(const T &alpha = T(1)) : __alpha(alpha) {}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(), __alpha);
	}

#else

	__host__ __device__
	__DLinear(const T &alpha = T(1)) : __alpha(alpha) {}
	
	__host__ __device__
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(), __alpha);
	}

#endif

};

// ReLU activation class
template <class T>
class __DReLU : public Activation <T> {
public:

#ifndef ZHP_CUDA

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) { 
				return (x[i] > 0) ? 1 : 0;
			}
		);
	}

#else

	__host__ __device__
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x] __host__ __device__ (size_t i) {
				return (x[i] > 0) ? 1 : 0;
			}
		);
	}

#endif

};

// Leaky ReLU activation class
template <class T>
class __DLeakyReLU : public Activation <T> {
	T	__alpha;
public:
	__DLeakyReLU(const T &alpha = 1) : __alpha(alpha) {}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				return (x[i] < 0) ? __alpha : 1;
			}
		);
	}
};

// Sigmoid activation class
template <class T>
class __DSigmoid : public Activation <T> {
public:

#ifndef ZHP_CUDA

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				T tmp = 1.0/(1.0 + exp(-x[i]));

				return tmp * (T (1.0) - tmp);
			}
		);
	}

#else

	__host__ __device__
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x] __host__ __device__ (size_t i) {
				T tmp = 1.0/(1.0 + exp(-x[i]));

				return tmp * (T (1.0) - tmp);
			}
		);
	}

#endif

};

// Scaled Sigmoid activation class
template <class T>
class __DScaledSigmoid : public Activation <T> {
	T	__alpha;
public:
	__DScaledSigmoid(const T &alpha) : __alpha(alpha) {}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(), [&](size_t i) {return
				__d_scaled_sigmoid(x[i], __alpha);});
	}
};

// Probability activation class
template <class T>
class __DSoftmax : public Activation <T> {
public:
	Vector <T> compute(const Vector <T> &x) const {
		// Subtract by max for numerical stability
		T _max = x[0];
		for (size_t i = 1; i < x.size(); i++)
			_max = (_max > x[i]) ? _max : x[i];

		T _sum = 0;
		for (size_t i = 0; i < x.size(); i++)
			_sum += exp(x[i] - _max);

		return Vector <T> (x.size(),
			[&](size_t i) {
				return exp(x[i] - _max)
					* (_sum - exp(x[i] - _max))
					/ (_sum * _sum);
			}
		);
	}
};

}

}

#endif
