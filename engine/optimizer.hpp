#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// Engine headers
#include <dataset.hpp>
#include <gradient.hpp>

namespace zhetapi {

namespace ml {

// Optimizer class
template <class T>
class Optimizer {
protected:
	// Cached
	Vector <T> *	__a		= nullptr;
	Vector <T> *	__z		= nullptr;

	T		__eta		= 0;

	size_t		__size		= 0;

	bool		__switch	= false;

	// Functions
	Optimizer(T);
	
	virtual Matrix <T> *raw_gradient(
			Layer <T> *,
			size_t,
			const Vector <T> &,
			const Vector <T> &,
			Erf <T> *);
	
	virtual Matrix <T> *raw_batch_gradient(
			Layer <T> *,
			size_t,
			const DataSet <T> &,
			const DataSet <T> &,
			Erf <T> *);
	
	virtual Matrix <T> *raw_multithreaded_batch_gradient(
			Layer <T> *,
			size_t,
			const DataSet <T> &,
			const DataSet <T> &,
			Erf <T> *,
			size_t);

	virtual Matrix <T> *update(
			Matrix <T> *,
			size_t) = 0;
public:
	virtual ~Optimizer();

	void set_learning_rate(T);

	Matrix <T> *gradient(
			Layer <T> *,
			size_t,
			const Vector <T> &,
			const Vector <T> &,
			Erf <T> *);

	Matrix <T> *batch_gradient(
			Layer <T> *,
			size_t,
			const DataSet <T> &,
			const DataSet <T> &,
			Erf <T> *);
	
	Matrix <T> *multithreaded_batch_gradient(
			Layer <T> *,
			size_t,
			const DataSet <T> &,
			const DataSet <T> &,
			Erf <T> *,
			size_t = 1);
};

template <class T>
Optimizer <T> ::Optimizer(T lr) : __eta(lr) {}

template <class T>
Optimizer <T> ::~Optimizer() 
{
	delete[] __a;
	delete[] __z;
}

template <class T>
void Optimizer <T> ::set_learning_rate(T lr)
{
	__eta = lr;
}

template <class T>
Matrix <T> *Optimizer <T> ::raw_gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
{
	if (size != __size) {
		delete[] __a;
		delete[] __z;

		__size = size;

		__a = new Vector <T> [__size + 1]; 
		__z = new Vector <T> [__size];

		__switch = true;
	} else {
		__switch = false;
	}
	
	return simple_gradient(
			layers,
			size,
			__a,
			__z,
			in,
			out,
			cost);
}

template <class T>
Matrix <T> *Optimizer <T> ::raw_batch_gradient(
			Layer <T> *layers,
			size_t size,
			const DataSet <T> &ins,
			const DataSet <T> &outs,
			Erf <T> *cost)
{
	if (size != __size) {
		delete[] __a;
		delete[] __z;

		__size = size;

		__a = new Vector <T> [__size + 1]; 
		__z = new Vector <T> [__size]; 

		__switch = true;
	} else {
		__switch = false;
	}
	
	return simple_batch_gradient(
			layers,
			size,
			__a,
			__z,
			ins,
			outs,
			cost);
}

template <class T>
Matrix <T> *Optimizer <T> ::raw_multithreaded_batch_gradient(
			Layer <T> *layers,
			size_t size,
			const DataSet <T> &ins,
			const DataSet <T> &outs,
			Erf <T> *cost,
			size_t threads)
{
	if (size != __size) {
		// Although changing __a and __z are not really required for the
		// multithreaded algorithm, we do it in case of future requests,
		// in which the size stays the same.
		delete[] __a;
		delete[] __z;

		__size = size;

		__a = new Vector <T> [__size + 1]; 
		__z = new Vector <T> [__size]; 

		__switch = true;
	} else {
		__switch = false;
	}
	
	return simple_multithreaded_batch_gradient(
			layers,
			size,
			__a,
			__z,
			ins,
			outs,
			cost,
			threads);
}

template <class T>
Matrix <T> *Optimizer <T> ::gradient(
			Layer <T> *layers,
			size_t size,
			const Vector <T> &in,
			const Vector <T> &out,
			Erf <T> *cost)
{

	return update(raw_gradient(
			layers,
			size,
			in,
			out,
			cost), size);
}

template <class T>
Matrix <T> *Optimizer <T> ::batch_gradient(
			Layer <T> *layers,
			size_t size,
			const DataSet <T> &ins,
			const DataSet <T> &outs,
			Erf <T> *cost)
{
	return update(raw_batch_gradient(
			layers,
			size,
			ins,
			outs,
			cost), size);
}

template <class T>
Matrix <T> *Optimizer <T> ::multithreaded_batch_gradient(
			Layer <T> *layers,
			size_t size,
			const DataSet <T> &ins,
			const DataSet <T> &outs,
			Erf <T> *cost,
			size_t threads)
{
	return update(raw_multithreaded_batch_gradient(
			layers,
			size,
			ins,
			outs,
			cost,
			threads), size);
}

}

}

#endif
