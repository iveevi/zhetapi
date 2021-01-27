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
struct RandomInitializer {
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

	std::function <T ()>	__initializer;

	void clear();
public:
	// Memory operations
	Layer();
	Layer(size_t, Activation <T> *,
			std::function <T ()> = RandomInitializer <T> ());

	Layer(const Layer &);

	Layer &operator=(const Layer &);

	~Layer();

	Matrix <T> &mat() {return __mat;}

	// Getters and setters
	size_t get_fan_in() const;
	size_t get_fan_out() const;

	void set_fan_in(size_t);

	// Initialize
	void initialize();

	// Computation
	Vector <T> forward_propogate(const Vector <T> &);

	void forward_propogate(Vector <T> &, Vector <T> &);
	
	void apply_gradient(const Matrix <T> &);

	// Diagnosing methods
	void diagnose() const;

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

	template <class U>
	friend Layer <U> operator-(const Layer <U> &, const Matrix <U> &);
};

// Memory operations
template <class T>
Layer <T> ::Layer() {}

template <class T>
Layer <T> ::Layer(size_t fan_out, Activation <T> *act,
		std::function <T ()> init) :
		__fan_out(fan_out),
		__act(act),
		__initializer(RandomInitializer<T> ())
{
	std::cout << "init() = " << __initializer() << std::endl;
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

		if (other.__act) {
			__act = other.__act->copy();
			__dact = __act->derivative();
		}
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
size_t Layer <T> ::get_fan_in() const
{
	return __fan_in;
}

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

using namespace std;

// Computation
template <class T>
inline Vector <T> Layer <T> ::forward_propogate(const Vector <T> &in)
{
	return __act->compute(__mat * in.append_above(T (1)));
}

template <class T>
inline void Layer <T> ::forward_propogate(Vector <T> &in1, Vector <T> &in2)
{
	in2 = __mat * in1.append_above(T (1));
	in1 = __act->compute(in2);
}

template <class T>
inline void Layer <T> ::apply_gradient(const Matrix <T> &J)
{
	__mat -= J;
}

// Diagnosing
template <class T>
void Layer <T> ::diagnose() const
{
	using namespace std;
	cout << "__mat = " << __mat << endl;
	for (size_t i = 0; i < __mat.get_rows(); i++) {
		for (size_t j = 0; j < __mat.get_cols(); j++) {
			cout << "__mat[" << i << "][" << j << "] = " << __mat[i][j] << endl;
			if (__mat[i][j] == 0)
				cout << "\t[!] element " << endl;
		}
	}
}

// Friends
template <class T>
Layer <T> operator-(const Layer <T> &L, const Matrix <T> &M)
{
	Layer <T> out = L;

	out.__mat -= M;

	return out;
}

}

}

#endif
