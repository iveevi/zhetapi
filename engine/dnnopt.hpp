#ifndef DNN_OPT_H_
#define DNN_OPT_H_

// Engine headers
#include "dataset.hpp"
#include "gradient.hpp"

namespace zhetapi {

template <class T>
class DnnOpt {
protected:
	// Cached
	Vector <T> *	__a		= nullptr;
	Vector <T> *	__z		= nullptr;

	T		__eta		= 0;

	size_t		__size		= 0;

	bool		__switch	= false;

	// Functions
	DnnOpt(T);

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

	virtual Matrix <T> *update(
			Matrix <T> *,
			size_t) = 0;
public:
	virtual ~DnnOpt();

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
};

template <class T>
DnnOpt <T> ::DnnOpt(T lr) : __eta(lr) {}

template <class T>
DnnOpt <T> ::~DnnOpt()
{
	delete[] __a;
	delete[] __z;
}

template <class T>
void DnnOpt <T> ::set_learning_rate(T lr)
{
	__eta = lr;
}

template <class T>
Matrix <T> *DnnOpt <T> ::raw_gradient(
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
Matrix <T> *DnnOpt <T> ::raw_batch_gradient(
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
Matrix <T> *DnnOpt <T> ::gradient(
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
Matrix <T> *DnnOpt <T> ::batch_gradient(
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

}

}

#endif
