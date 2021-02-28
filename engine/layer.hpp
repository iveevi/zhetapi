#ifndef LAYERS_H_
#define LAYERS_H_

// JSON library
#include <json/json.hpp>

// Engine headers
#include <matrix.hpp>
#include <activation.hpp>

#include <std/interval.hpp>

namespace zhetapi {

namespace ml {

// Aliases
using utility::Interval;

template <class T>
class Erf;

// Move a separate file with standard initializers
template <class T>
struct RandomInitializer {
	T operator()() {
		return T (0.5 - rand()/((double) RAND_MAX));
	}
};

template <class T = double>
class Layer {
	size_t			__fan_in	= 0;
	size_t			__fan_out	= 0;

	Matrix <T>		__mat		= Matrix <T> ();

	Activation <T> *	__act		= nullptr;
	Activation <T> *	__dact		= nullptr;

	std::function <T ()>	__initializer;

	long double		__dropout	= 0;

	void clear();

	static Interval <1>	__unit;
public:
	// Memory operations
	Layer();
	Layer(size_t, Activation <T> *,
			std::function <T ()> = RandomInitializer <T> (),
			long double = 0.0);

	Layer(const Layer &);

	Layer &operator=(const Layer &);

	~Layer();

	Matrix <T> &mat() {return __mat;}

	// Getters and setters
	size_t get_fan_in() const;
	size_t get_fan_out() const;

	void set_fan_in(size_t);

	// Read and write
	void write(std::ofstream &) const;
	void read(std::ifstream &);

	// Initialize
	void initialize();

	// Computation
	Vector <T> forward_propogate(const Vector <T> &);

	// Computation with dropout
	void forward_propogate(Vector <T> &, Vector <T> &);
	
	void apply_gradient(const Matrix <T> &);

	// Diagnosing methods
	void diagnose() const;
	void print() const;

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

	Layer <T> &operator+=(const Matrix <T> &);

	template <class U>
	friend Layer <U> operator-(const Layer <U> &, const Matrix <U> &);
	
	template <class U>
	friend Layer <U> operator+(const Layer <U> &, const Matrix <U> &);
};

// Static variables
template <class T>
Interval <1> Layer <T> ::__unit(1.0L);

// Memory operations
template <class T>
Layer <T> ::Layer() {}

template <class T>
Layer <T> ::Layer(size_t fan_out, Activation <T> *act,
		std::function <T ()> init,
		long double dropout) :
		__fan_out(fan_out),
		__act(act),
		__initializer(RandomInitializer <T> ()),
		__dropout(dropout)
{
	__dact = __act->derivative();
}

template <class T>
Layer <T> ::Layer(const Layer <T> &other) :
		__fan_in(other.__fan_in),
		__fan_out(other.__fan_out),
		__act(other.__act->copy()),
		__mat(other.__mat),
		__initializer(other.__initializer),
		__dropout(other.__dropout)
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

		__mat = other.__mat;

		__initializer = other.__initializer;

		__dropout = other.__dropout;

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

// Reading and writing
template <class T>
void Layer <T> ::write(std::ofstream &fout) const
{
	size_t r = __mat.get_rows();
	size_t c = __mat.get_cols();

	fout.write((char *) &r, sizeof(size_t));
	fout.write((char *) &c, sizeof(size_t));

	__mat.write(fout);
	__act->write(fout);
}

template <class T>
void Layer <T> ::read(std::ifstream &fin)
{
	size_t r;
	size_t c;

	fin.read((char *) &r, sizeof(size_t));
	fin.read((char *) &c, sizeof(size_t));

	__mat = Matrix <T> (r, c, T(0));

	__mat.read(fin);

	__act = Activation <T> ::load(fin);
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
	return __act->compute(__mat * in.append_above(T (1)));
}

template <class T>
inline void Layer <T> ::forward_propogate(Vector <T> &in1, Vector <T> &in2)
{
	in2 = apt_and_mult(__mat, in1);
	in1 = __act->compute(in2);

	// Apply dropout (only if necessary)
	if (__dropout > 0)
		in1.nullify(__dropout, __unit);
}

template <class T>
inline void Layer <T> ::apply_gradient(const Matrix <T> &J)
{
	__mat += J;
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

template <class T>
void Layer <T> ::print() const
{
	std::cout << "W = " << __mat << std::endl;
}

template <class T>
Layer <T> &Layer <T> ::operator+=(const Matrix <T> &M)
{
	__mat += M;

	return *this;
}

// Friends
template <class T>
Layer <T> operator+(const Layer <T> &L, const Matrix <T> &M)
{
	Layer <T> out = L;

	out.__mat += M;

	return out;
}

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
