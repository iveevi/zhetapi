#ifndef STD_OPTIMIZERS_H_
#define STD_OPTIMIZERS_H_

// Engine headers
#include <optimizer.hpp>

namespace zhetapi {

namespace ml {

// TODO: Cache a and z vectors in the class itself
// In general cache as much data as possible
// Instead of creating a new gradient, modify the one passed

/*
 * This header contains the standard optimizers.
 *
 * Preferably use one optimizer per neural network.
 */

template <class T>
class SGD : public Optimizer <T> {
public:
	SGD(T eta = 0.001) : Optimizer <T> (eta) {}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		this->register_size(size);
		for (size_t i = 0; i < size; i++)
			J[i] *= -1 * this->_eta;
		
		return J;
	}
};

/*
 * TODO: Cache J in the optimizer itself so that we do not need to repeatedly
 * allocate and free the gradient.
 */
template <class T>
class Momentum : public Optimizer <T> {
	T		_mu	= 0;

	Matrix <T> *	_M	= nullptr;
public:
	Momentum(T eta = 0.001, T mu = 0.9)
			: Optimizer <T> (eta),
			_mu(mu) {}

	~Momentum() {
		delete[] _M;
	}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		this->register_size(size);
		if (this->_switch) {
			delete[] _M;

			_M = new Matrix <T> [size];
		}
		
		Matrix <T> *Jo = new Matrix <T> [size];
		for (size_t i = 0; i < size; i++) {
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (_M[i].get_dimensions() != odim)
				_M[i] = Matrix <T> (odim.first, odim.second, T(0));
			
			Jo[i] = _M[i] = _mu * _M[i] - this->_eta * J[i];
		}

		delete[] J;

		return Jo;
	}
};

// TODO: Fix the whole raw gradient requirement with Nesterov

// TODO: Inherit Nesterov from Momentum
/*
template <class T>
class Nesterov : public Optimizer <T> {
	T		_mu;

	Matrix <T> *	_M = nullptr;
public:
	Nesterov(T eta = 0.001, T mu = 0.9)
			: Optimizer <T> (eta),
			_mu(mu) {}

	~Nesterov() {
		delete[] _M;
	}

	Layer <T> *adjusted(Layer <T> *layers, size_t size)
	{
		Layer <T> *adj = new Layer <T> [size];
		
		for (size_t i = 0; i < size; i++)
			adj[i] = layers[i];

		if (size != this->_size)
			return adj;

		for (size_t i = 0; i < size; i++) {
			size_t rs = layers[i].get_fan_out();
			size_t cs = layers[i].get_fan_in() + 1;

			if (_M[i].get_dimensions() != std::make_pair(rs, cs))
				_M[i] = Matrix <T> (rs, cs, T(0));

			adj[i] += _mu * _M[i];
		}

		return adj;
	}

	Matrix <T> *raw_gradient(
				Layer <T> *layers,
				size_t size,
				const Vector <T> &in,
				const Vector <T> &out,
				Erf <T> *cost) override
	{
		Layer <T> *adj = adjusted(layers, size);

		Matrix <T> *Delta = Optimizer <T> ::raw_gradient(
				adj,
				size,
				in,
				out,
				cost);

		delete[] adj;

		return Delta;
	}

	Matrix <T> *raw_batch_gradient(
				Layer <T> *layers,
				size_t size,
				const DataSet <T> &ins,
				const DataSet <T> &outs,
				Erf <T> *cost) override
	{
		Layer <T> *adj = adjusted(layers, size);

		Matrix <T> *Delta = Optimizer <T> ::raw_batch_gradient(
				adj,
				size,
				ins,
				outs,
				cost);

		delete[] adj;

		return Delta;
	}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		this->register_size(size);
		if (this->_switch) {
			delete[] _M;

			_M = new Matrix <T> [size];
		}
		
		Matrix <T> *Jo = new Matrix <T> [size];
		for (size_t i = 0; i < size; i++) {
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (_M[i].get_dimensions() != odim)
				_M[i] = Matrix <T> (odim.first, odim.second, T(0));
			
			Jo[i] = _M[i] = _mu * _M[i] - this->_eta * J[i];
		}

		delete[] J;

		return Jo;
	}
}; */

template <class T>
class AdaGrad : public Optimizer <T> {
	Matrix <T> *	_S = nullptr;
public:
	AdaGrad(T eta = 0.001)
			: Optimizer <T> (eta) {}

	~AdaGrad() {
		delete[] _S;
	}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		this->register_size(size);
		if (this->_switch) {
			delete[] _S;

			_S = new Matrix <T> [size];
		}

		Matrix <T> tmp;
		for (size_t i = 0; i < size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (_S[i].get_dimensions() != odim)
				_S[i] = Matrix <T> (odim.first, odim.second, T(0));

			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			_S[i] = _S[i] + shur(J[i], J[i]);

			tmp = _S[i] + e;

			tmp.pow(0.5);

			J[i] = inv_shur((-1 * this->_eta) * J[i], tmp);
		}

		return J;
	}

	static const T epsilon;
};

template <class T>
const T AdaGrad <T> ::epsilon = 1e-10;

template <class T>
class RMSProp : public Optimizer <T> {
	T		_beta;

	Matrix <T> *	_S = nullptr;
public:
	RMSProp(T eta = 0.001, T beta = 0.9)
			: Optimizer <T> (eta),
			_beta(beta) {}

	~RMSProp() {
		delete[] _S;
	}

	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		this->register_size(size);
		if (this->_switch) {
			delete[] _S;

			_S = new Matrix <T> [size];
		}

		Matrix <T> tmp;
		for (size_t i = 0; i < size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			if (_S[i].get_dimensions() != odim)
				_S[i] = Matrix <T> (odim.first, odim.second, T(0));

			// TODO: Replace with element wise addition operation
			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			_S[i] = _beta * _S[i] + (1 - _beta) * shur(J[i], J[i]);

			tmp = _S[i] + e;

			tmp.pow(0.5);

			J[i] = inv_shur((-1 * this->_eta) * J[i], tmp);
		}

		return J;
	}
	
	static const T epsilon;
};

template <class T>
const T RMSProp <T> ::epsilon = 1e-10;

template <class T>
class Adam : public Optimizer <T> {
	T		_beta1;
	T		_beta2;

	Matrix <T> *	_M	= nullptr;
	Matrix <T> *	_Mh	= nullptr;
	Matrix <T> *	_S	= nullptr;
	Matrix <T> *	_Sh	= nullptr;

	size_t		_iter	= 1;
public:
	Adam(T eta = 0.001, T beta1 = 0.9, T beta2 = 0.999)
			: Optimizer <T> (eta),
			_beta1(beta1),
			_beta2(beta2) {}

	~Adam() {
		delete[] _M;
		delete[] _Mh;
		delete[] _S;
		delete[] _Sh;
	}
	
	// TODO: reset if the layers pointer is different
	// from the cashed pointer
	Matrix <T> *update(Matrix <T> *J, size_t size)
	{
		this->register_size(size);
		using namespace std;
		if (this->_switch)
			reset(size);

		Matrix <T> tmp;
		for (size_t i = 0; i < size; i++) {
			// Correct matrix size if necessary
			std::pair <size_t, size_t> odim = J[i].get_dimensions();

			// Resize the matrices if necessary
			resize(odim, i);

			Matrix <T> e = Matrix <T> (odim.first, odim.second, epsilon);

			_M[i] = _beta1 * _M[i] - (1 - _beta1) * J[i];
			_S[i] = _beta2 * _S[i] + (1 - _beta2) * shur(J[i], J[i]);

			_Mh[i] = _M[i] / T(1 - pow(_beta1, _iter));
			_Sh[i] = _S[i] / T(1 - pow(_beta2, _iter));

			tmp = _Sh[i] + e;

			tmp.pow(0.5);

			J[i] = inv_shur(this->_eta * _Mh[i], tmp);
		}

		_iter++;

		return J;
	}
	
	void reset(size_t size) {
		delete[] _M;
		delete[] _Mh;
		delete[] _S;
		delete[] _Sh;

		_iter = 1;

		_M = new Matrix <T> [size];
		_Mh = new Matrix <T> [size];
		_S = new Matrix <T> [size];
		_Sh = new Matrix <T> [size];
	}

	inline void resize(const std::pair <size_t, size_t> &odim, size_t i) {
		if (_M[i].get_dimensions() != odim)
			_M[i] = Matrix <T> (odim.first, odim.second, T(0));
		if (_Mh[i].get_dimensions() != odim)
			_Mh[i] = Matrix <T> (odim.first, odim.second, T(0));
		if (_S[i].get_dimensions() != odim)
			_S[i] = Matrix <T> (odim.first, odim.second, T(0));
		if (_Sh[i].get_dimensions() != odim)
			_Sh[i] = Matrix <T> (odim.first, odim.second, T(0));
	}

	static const T epsilon;
};

template <class T>
const T Adam <T> ::epsilon = 1e-10;

}

}

#endif
