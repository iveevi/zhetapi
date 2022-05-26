#ifndef ZHETAPI_AUTOGRAD_ML_H_
#define ZHETAPI_AUTOGRAD_ML_H_

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

	// Weight matrix
	Matrix <float>			_w;

	// Bias
	Matrix <float>			_b;

	// Private constructor
	_kdense (size_t isize, size_t osize, Matrix <float> w, Matrix <float> b)
			: _function(1), _isize (isize), _osize (osize),
			_w (w), _b (b) {}

	// Static random number generator
	static utility::Interval <1>	_rng;

	static float rgen(size_t) {
		return _rng();
	}
public:
	_kdense(size_t isize, size_t osize)
			: _function(1), _isize(isize), _osize(osize),
		_w(isize, osize, rgen), _b(osize, 1, rgen) {}

	// Copy overload
	_function *copy() const override {
		return new _kdense(_isize, _osize, _w, _b);
	}

	// Forward pass
	Constant compute(const Input &ins) const override {
		// Convert first argument into a matrix
		Matrix <float> x(ins[0], _isize, 1);
		return _w * x + _b;
	}

	// Machine learning functions
	virtual Input gradient(const Constant &igrad, const Input &ins) const override {
		// igrad is the gradient of the output of the
		// function wrt to the desired function
		std::cout << "igrad: " << igrad << std::endl;
		return {_w.transpose()};
	}

	// Summary of the function
	std::string summary() const override {
		return "KDENSE (" + std::to_string(_isize) + " x " + std::to_string(_osize) + ")";
	}
};

class _dense : public ISeq {
public:
	_dense(size_t isize, size_t osize)
		: ISeq(new _kdense(isize, osize), 1) {}
};

// Dense layer factory
inline Function dense(size_t isize, size_t osize)
{
	return new_ <_dense> (isize, osize);
}

}

}

}

#endif
