#ifndef LAYERS_H_
#define LAYERS_H_

// Engine headers
#include <matrix.hpp>
#include <activation.hpp>

namespace zhetapi {

namespace ml {

template <class T>
class Erf;

// Move a separate file with standard initializers
template <class T>
struct random_initializer {
	T operator()() {
		return T (rand() / ((double) RAND_MAX));
	}
};

template <class T>
class Layer {
	size_t			__fan_in = 0;
	size_t			__fan_out = 0;

	Matrix <T>		__mat = Matrix <T> ();

	Activation <T> *	__act = nullptr;
	Activation <T> *	__dact = nullptr;

	std::function <T ()>	__initializer = random_initializer <T> ();

	void clear();
public:
	// Memory operations
	Layer();
	Layer(size_t, Activation <T> *);

	Layer(const Layer &);

	Layer &operator=(const Layer &);

	~Layer();

	// Getters and setters
	size_t get_fan_out() const;

	void set_fan_in(size_t);

	// Initialize
	void initialize();

	// Computation
	Vector <T> forward_propogate(const Vector <T> &);

	void forward_propogate(Vector <T> &, Vector <T> &);

	// Friend functions
	template <class U>
	friend Vector <U> simple_compute_cached(
		Layer <U> *,
		size_t,
		Vector <U> *,
		Vector <U> *,
		const Vector <U> &
	);
	
	template <class U>
	friend Matrix <U> *simple_gradient(
		Layer <U> *,
		size_t,
		Vector <U> *,
		Vector <U> *,
		const Vector <U> &,
		const Vector <U> &,
		Erf <U> *
	);
};

// Memory operations
template <class T>
Layer <T> ::Layer() {}

template <class T>
Layer <T> ::Layer(size_t fan_out, Activation <T> *act) :
		__fan_out(fan_out),
		__act(act)
{
	__dact = __act->derivative();
}

template <class T>
Layer <T> ::Layer(const Layer <T> &other) :
		__fan_in(other.__fan_in),
		__fan_out(other.__fan_out),
		__act(other.__act->copy()),
		__mat(other.__mat)
{
	__dact = __act->derivative();
}

template <class T>
Layer <T> &Layer <T> ::operator=(const Layer <T> &other)
{
	if (this != &other) {
		clear();

		__fan_in = other.__fan_in;
		__fan_out = other.__fan_out;

		__act = other.__act->copy();
		__dact = __act->derivative();

		__mat = other.__mat;
	}

	return *this;
}

template <class T>
Layer <T> ::~Layer()
{
	clear();
}

template <class T>
void Layer <T> ::clear()
{
	if (__act)
		delete __act;

	if (__dact)
		delete __dact;
}

// Getters and setters
template <class T>
size_t Layer <T> ::get_fan_out() const
{
	return __fan_out;
}

template <class T>
void Layer <T> ::set_fan_in(size_t fan_in)
{
	__fan_in = fan_in;
	
	if (__fan_in * __fan_out > 0)
		__mat = Matrix <T> (__fan_out, __fan_in + 1);
}

// Initializer
template <class T>
void Layer <T> ::initialize()
{
	__mat.randomize(__initializer);
}

// Computation
template <class T>
inline Vector <T> Layer <T> ::forward_propogate(const Vector <T> &in)
{
	// std::cout << "in1 = " << in << std::endl;
	std::cout << "apped = " << in.append_above(T (1)) << std::endl;
	std::cout << "prv = " << __mat * in.append_above(T (1)) << std::endl;
	// std::cout << "in2 = " << __mat * in.append_above(T (1)) << std::endl;

	//std::cout << "\t__mat = " << __mat << std::endl;
	return __act->compute(__mat * in.append_above(T (1)));
}

template <class T>
inline void Layer <T> ::forward_propogate(Vector <T> &in1, Vector <T> &in2)
{
	std::cout << "in1 = " << in1 << std::endl;
	// std::cout << "\t__mat = " << __mat << std::endl;

	in2 = __mat * in1.append_above(T (1));
	// std::cout << "in2 = " << in2 << std::endl;

	in1 = __act->compute(in2);
}

}

}

#endif
