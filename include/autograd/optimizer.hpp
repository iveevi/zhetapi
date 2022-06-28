#include "function.hpp"

namespace zhetapi {

namespace autograd {

namespace ml {

// Interface for optimizers
class _optimizer {
protected:
	size_t _parameters = 0;
public:
	float alpha;

	_optimizer(size_t parameters, float alpha_ = 0.001)
		: _parameters(parameters), alpha(alpha_) {}

	// Computation
	virtual void optimize(_function::GradientQueue &) = 0;

	void operator()(_function::GradientQueue &gq) {
		optimize(gq);
	}
};

// Stochastic gradient descent
struct SGD : public _optimizer {
	SGD(size_t parameters, float alpha)
		: _optimizer(parameters, alpha) {}

	// Optimize
	void optimize(_function::GradientQueue &gq) override {
		for (auto &g : gq)
			g *= -alpha;
	}
};

// Momentum
class Momentum : public _optimizer {
	_function::GradientQueue _v;
public:
	float mu;

	Momentum(size_t parameters, float alpha, float mu_ = 0.9)
		: _optimizer(parameters, alpha), mu(mu_) {
		_v.resize(parameters);
	}

	// Optimize
	void optimize(_function::GradientQueue &gq) override {
		// TODO: assert that gq.size() == _v.size()
		for (size_t i = 0; i < _parameters; i++) {
			if (_v[i].shape() != gq[i].shape())
				_v[i] = Constant(gq[i].shape(), 0);

			_v[i] = mu * _v[i] - alpha * gq[i];
			gq[i] = _v[i];
		}
	}
};

// RMSprop
class RMSprop : public _optimizer {
	_function::GradientQueue _v;
public:
	float beta;

	RMSprop(size_t parameters, float alpha, float beta_ = 0.9)
		: _optimizer(parameters, alpha), beta(beta_) {
		_v.resize(parameters);
	}

	// Optimize
	void optimize(_function::GradientQueue &gq) override {
		// TODO: assert that gq.size() == _v.size()
		for (size_t i = 0; i < _parameters; i++) {
			if (_v[i].shape() != gq[i].shape())
				_v[i] = Constant(gq[i].shape(), 0);

			_v[i] = beta * _v[i] + (1 - beta) * gq[i] * gq[i];
			gq[i] = -alpha * gq[i] / _v[i].transform(
				[](float x) {
					return std::sqrt(x) + 1e-10;
				}
			);
		}
	}
};

// Adam
class Adam : public _optimizer {
	_function::GradientQueue _v, _m;
	size_t _iter = 1;
public:
	float beta1, beta2;

	Adam(size_t parameters, float alpha, float beta1_ = 0.9, float beta2_ = 0.999)
		: _optimizer(parameters, alpha), beta1(beta1_), beta2(beta2_) {
		_v.resize(parameters);
		_m.resize(parameters);
	}

	// Reset iteration
	void reset() {
		_iter = 1;
	}

	// Optimize
	void optimize(_function::GradientQueue &gq) override {
		// TODO: assert that gq.size() == _v.size()
		for (size_t i = 0; i < _parameters; i++) {
			if (_v[i].shape() != gq[i].shape())
				_v[i] = Constant(gq[i].shape(), 0);
			if (_m[i].shape() != gq[i].shape())
				_m[i] = Constant(gq[i].shape(), 0);

			_v[i] = beta1 * _v[i] - (1 - beta1) * gq[i];
			_m[i] = beta2 * _m[i] + (1 - beta2) * gq[i] * gq[i];

			auto _vh = _v[i]/float(1 - std::pow(beta1, _iter));
			auto _mh = _m[i].transform(
				[&](float x) -> float{
					return std::sqrt(1e-10 + x/float(1.0 - std::pow(beta2, _iter)));
				}
			);

			gq[i] = alpha * _vh / _mh;
		}

		_iter++;
	}
};

}

}

}
