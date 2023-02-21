#ifndef ZHETAPI_AUTOGRAD_ML_H_
#define ZHETAPI_AUTOGRAD_ML_H_

// Standard headers
#include <iomanip>
#include <random>

// Library headers
#include "../matrix.hpp"
#include "../vector.hpp"
#include "../std/interval.hpp"
#include "autograd.hpp"
#include "cpu_kernels.hpp"

namespace zhetapi {

namespace autograd {

namespace ml {

class _kdense : public _function {
	// Input and output shape
	size_t				_isize;
	size_t				_osize;
	std::string			_init;
	float				m_dropout;

	// Weight matrix
	Matrix <float>			_w;

	// Bias
	Matrix <float>			_b;

	// Cached resources
	Matrix <float>			m_output;

	// Static random number generator
	static utility::Interval <1>		_rng;
public:
	_kdense(size_t isize, size_t osize, const std::string &initializer = "xavier")
			: _function(1), _isize(isize), _osize(osize), m_output(osize, 1)
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
		else if (_init == "debug")
			lambda = [&](size_t i) { return 1.0f; };
		else
			lambda = [&](size_t i) { return 0.0f; };

		_w = Matrix <float> (_osize, _isize, lambda);
		_b = Matrix <float> (_osize, 1, lambda);
	}

	// Forward pass
	Constant compute(const Input &ins) override {
		// TODO: check if batching...
		// Convert first argument into a matrix
		detail::autograd::fma_matrix_vector(
			m_output.data(), _w.data(),
			_b.data(), ins[0].data(),
			_osize, _isize
		);

		return m_output;
	}

	// Machine learning functions
	virtual Gradient gradient(const Input &ins, const Input &igrads) override {
		// igrad is the gradient of the output of the
		// function wrt to the desired function
		Matrix <float>			igrad(_isize, 1);
		Matrix <float>			wgrad(_osize, _isize);
		Matrix <float>			bgrad(_osize, 1);

		detail::autograd::mul_vector_vector_transpose(
			wgrad.data(), igrads[0].data(), ins[0].data(),
			_osize, _isize
		);

		// TODO: Copy and computation in parallel?
		detail::autograd::mul_matrix_transpose_vector(
			igrad.data(), _w.data(), igrads[0].data(),
			_isize, _osize
		);

		bgrad.copy(igrads[0]);

		// TODO: avoid the need to copy... reduce required allocations
		// Debug copy issues when using persistent gradient storage...
		Gradient gradient;
		gradient.igrads = igrads,
		gradient.grads = { wgrad, bgrad };
		return gradient;
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
	}

	// Info about parameters
	virtual int parameters() const override {
		return 2;
	}

	virtual int tunable_parameters() const override {
		return _w.size() + _b.size();
	}

	// Method table
	std::pair <_function *, const MethodTable &> method_table() override {
		static const MethodTable _map {
			{"dropout", [](_function  *f, const Arguments &args) {
				_kdense *kf = dynamic_cast <_kdense *> (f);

				assert(kf);
				if (args.size() > 0)
					kf->m_dropout = std::get <float> (args[0]);

				return kf->m_dropout;
			}}
		};

		return {this, _map};
	}

	// Summary of the function
	std::string summary() const override {
		std::ostringstream oss;
		oss << "DENSE(" << _isize << " x " << _osize;
		if (m_dropout > 0)
			oss << ", dropout = " << std::setprecision(2) << m_dropout;
		oss << ", " << _init << ")";
		return oss.str();
	}
};

class _dense : public ISeq {
public:
	_dense(size_t isize, size_t osize, const std::string &initializer = "xavier")
		: ISeq(new_ftn_ <_kdense> (isize, osize, initializer), 1) {}
};

// Dense layer factory
inline Function dense(size_t isize, size_t osize, const std::string &initializer = "xavier")
{
	return Function(new_ftn_ <_dense> (isize, osize, initializer));
}

}

}

}

#endif
