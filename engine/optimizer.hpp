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
	T		_eta		= 0;
	size_t		_size		= 0;
	bool		_switch	= false;

	// Functions
	Optimizer(T);
public:
	void register_size(size_t);
	void set_learning_rate(T);
	
	virtual Matrix <T> *update(
			Matrix <T> *,
			size_t) = 0;
};

template <class T>
Optimizer <T> ::Optimizer(T lr) : _eta(lr) {}

template <class T>
void Optimizer <T> ::register_size(size_t size)
{
	if (_size != size) {
		_size = size;
		_switch = true;
	} else {
		_switch = false;
	}
}

template <class T>
void Optimizer <T> ::set_learning_rate(T lr)
{
	_eta = lr;
}

}

}

#endif
