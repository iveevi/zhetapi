#ifndef ACTIVATION_DERIVATIVES_H_
#define ACTIVATION_DERIVATIVES_H_

// TODO: put back in std
namespace zhetapi {

namespace ml {

template <class T>
class _DLinear : public Activation <T> {
	T	_alpha;
public:
	__cuda_dual__
	explicit _DLinear(const T &alpha = T(1)) : _alpha(alpha) {}

	__cuda_dual__
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(), _alpha);
	}
};

// ReLU activation class
template <class T>
class _DReLU : public Activation <T> {
public:
	__cuda_dual__
	Vector <T> compute(const Vector <T> &x) const {
		T *arr = new T[x.size()];
		for (size_t i = 0; i < x.size(); i++) {
			arr[i] = (x[i] > 0) ? 1 : 0;
		}
		return Vector <T> (x.size(), arr, false);
	}
};

// Leaky ReLU activation class
template <class T>
class _DLeakyReLU : public Activation <T> {
	T	_alpha;
public:
	_DLeakyReLU(const T &alpha = 1) : _alpha(alpha) {}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[&](size_t i) {
				return (x[i] < 0) ? _alpha : 1;
			}
		);
	}
};

// Sigmoid activation class
template <class T>
class _DSigmoid : public Activation <T> {
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

	_host_ _device_
	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(),
			[x] _host_ _device_ (size_t i) {
				T tmp = 1.0/(1.0 + exp(-x[i]));

				return tmp * (T (1.0) - tmp);
			}
		);
	}

#endif

};

// Scaled Sigmoid activation class
template <class T>
class _DScaledSigmoid : public Activation <T> {
	T	_alpha;
public:
	_DScaledSigmoid(const T &alpha) : _alpha(alpha) {}

	Vector <T> compute(const Vector <T> &x) const {
		return Vector <T> (x.size(), [&](size_t i) {return
				_d_scaled_sigmoid(x[i], _alpha);});
	}
};

// Probability activation class
template <class T>
class _DSoftmax : public Activation <T> {
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
