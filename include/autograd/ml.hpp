#ifndef ZHETAPI_AUTOGRAD_ML_H_
#define ZHETAPI_AUTOGRAD_ML_H_

// Standard headers
#include <random>

// Library headers
#include "../matrix.hpp"
#include "../vector.hpp"
#include "../std/interval.hpp"
#include "autograd.hpp"

namespace zhetapi {

namespace autograd {

namespace ml {

class _kdense : public _function {
	// Input and output shape
	size_t				_isize;
	size_t				_osize;
	std::string			_init;

	// Weight matrix
	Matrix <float>			_w;

	// Bias
	Matrix <float>			_b;

	// Cached
	Matrix <float>			_cO;
	Matrix <float>			_cI;

	// Private constructor
	_kdense(size_t isize, size_t osize, Matrix <float> w, Matrix <float> b,
			const std::string &init)
			: _function(1), _isize(isize), _osize(osize),
			_w(w), _b(b), _init(init) {}

	// Static random number generator
	static utility::Interval <1>		_rng;
public:
	_kdense(size_t isize, size_t osize, const std::string &initializer = "xavier")
			: _function(1), _isize(isize), _osize(osize)
	{
		// Lower case initializer
		for (auto &c : initializer)
			_init += std::tolower(c);

		// Initializer
		std::function <float (size_t)> lambda = [](size_t) { return _rng(); };
	
		std::random_device rd;
		std::mt19937 gen(rd());

		std::normal_distribution <float> dist;

		int normal = 0;
		if (_init == "lecun") {
			dist = std::normal_distribution <float> (0, 1.0 / std::sqrt(isize));
			normal++;
		} else if (_init == "he") {
			dist = std::normal_distribution <float> (0, 2.0/std::sqrt(isize));
			normal++;
		} else if (_init == "xavier") {
			float avg = (isize + osize) / 2.0f;
			dist = std::normal_distribution <float> (0, 1.0/std::sqrt(avg));
			normal++;
		}

		if (normal)
			lambda = [&](size_t i) { return dist(gen); };

		// _w = Matrix <float> (_osize, _isize, lambda);
		// _b = Matrix <float> (_osize, 1, lambda);
		
		_w = Matrix <float> (_osize, _isize, 0);
		_b = Matrix <float> (_osize, 1, 0);
	}

	// Copy overload
	_function *copy() const override {
		return new _kdense(_isize, _osize, _w, _b, _init);
	}

	// Forward pass
	Constant compute(const Input &ins) override {
		// Convert first argument into a matrix
		_cI = Matrix <float> (ins[0], _isize, 1);
		_cO = _w * _cI + _b;
		return Constant(_cO);
	}

	// Machine learning functions
	virtual Gradient gradient(const Input &igrads) const override {
		// igrad is the gradient of the output of the
		// function wrt to the desired function
		Matrix <float> dO(igrads[0], _osize, 1);
		std::cout << "\ndO = " << dO << std::endl;
		Matrix <float> wgrad = dO * _cI.transpose();
		Matrix <float> bgrad = dO;
		Matrix <float> igrad = _w.transpose() * dO;

		return Gradient {
			.igrads = {igrad},
			.grads = {wgrad, bgrad}
		};
	}

	// Apply gradient
	virtual void update_parameters(GradientQueue &grads) override {
		// Convert first argument into a matrix
		Matrix <float> bgrad(grads.back(), _osize, 1);
		grads.pop_back();

		Matrix <float> wgrad(grads.back(), _osize, _isize);
		grads.pop_back();

		_w += wgrad;
		_b += bgrad;

		std::cout << "\ngrads:\n" << std::endl;
		std::cout << "wgrad = " << wgrad << std::endl;
		std::cout << "bgrad = " << bgrad << std::endl;
	}

	// Info about parameters
	virtual int parameters() const override {
		return 2;
	}

	virtual int tunable_parameters() const override {
		return _w.size() + _b.size();
	}

	// Summary of the function
	std::string summary() const override {
		return "KDENSE (" + std::to_string(_isize)
			+ " x " + std::to_string(_osize)
			+ ", " + _init + ")";
	}
};

class _dense : public ISeq {
public:
	_dense(size_t isize, size_t osize, const std::string &initializer = "xavier")
		: ISeq(new _kdense(isize, osize, initializer), 1) {}
};

// Dense layer factory
inline Function dense(size_t isize, size_t osize, const std::string &initializer = "xavier")
{
	return Function(new _dense(isize, osize, initializer));
}

}

}

}

#endif
