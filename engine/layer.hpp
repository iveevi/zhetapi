#ifndef LAYERS_H_
#define LAYERS_H_

#ifndef __AVR	// Does not support AVR

// JSON library
#include "json/json.hpp"

#endif		// Does not support AVR

// Engine headers
#include "matrix.hpp"
#include "activation.hpp"

#include "std/interval.hpp"
#include "std/initializers.hpp"

namespace zhetapi {

namespace ml {

// Aliases
using utility::Interval;

template <class T>
using InitFtn = AVR_SWITCH(
	T (*)(),
	std::function <T ()>
);

template <class T>
class Erf;

// FIXME: Dropout should not be active during inference phase
template <class T = double>
class Layer {
	size_t			_fan_in	= 0;
	size_t			_fan_out	= 0;

	Matrix <T>		_mat		= Matrix <T> ();

	Activation <T> *	_act		= nullptr;
	Activation <T> *	_dact		= nullptr;

	InitFtn	<T>		_initializer;

	// Dropout is off by default
	long double		_dropout	= 0;
	bool			_dp_enable	= false;

	void clear();

	static Interval <1>	_unit;
public:
	// Memory operations
	Layer();
	Layer(size_t, Activation <T> *,
			InitFtn <T> = AVR_SWITCH(
				RandomInitializer <T>,
				RandomInitializer <T> ()
			),
			long double = 0.0);

	Layer(const Layer &);

	Layer &operator=(const Layer &);

	~Layer();

	Matrix <T> &mat() {return _mat;}

	// Getters and setters
	size_t get_fan_in() const;
	size_t get_fan_out() const;

	void set_fan_in(size_t);
	
	// Enable and disable dropout
	void enable_dropout() const;
	void disable_dropout() const;

	// Read and write
	AVR_IGNORE(void write(std::ofstream &) const);
	AVR_IGNORE(void read(std::ifstream &));

	// Initialize
	void initialize();

	// Computation
	Vector <T> forward_propogate(const Vector <T> &);

	// Computation with dropout
	void forward_propogate(Vector <T> &, Vector <T> &);
	
	void apply_gradient(const Matrix <T> &);

	// Diagnosing methods
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
	friend Matrix <U> *jacobian_kernel(
		Layer <U> *,
		size_t,
		size_t,
		Vector <U> *,
		Vector <U> *,
		const Vector <U> &
	);

	template <class U>
	friend Matrix <U> *jacobian_kernel(
		Layer <U> *,
		size_t,
		size_t,
		Vector <U> *,
		Vector <U> *,
		const Vector <U> &,
		Vector <U> &
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
Interval <1> Layer <T> ::_unit(1.0L);

// Memory operations
template <class T>
Layer <T> ::Layer() {}

template <class T>
Layer <T> ::Layer(size_t fan_out, Activation <T> *act,
		InitFtn <T> init,
		long double dropout) :
		_fan_out(fan_out),
		_act(act),
		_initializer(init),
		_dropout(dropout)
{
	_dact = _act->derivative();
}

template <class T>
Layer <T> ::Layer(const Layer <T> &other) :
		_fan_in(other._fan_in),
		_fan_out(other._fan_out),
		_act(other._act->copy()),
		_mat(other._mat),
		_initializer(other._initializer),
		_dropout(other._dropout)
{
	_dact = _act->derivative();
}

template <class T>
Layer <T> &Layer <T> ::operator=(const Layer <T> &other)
{
	if (this != &other) {
		clear();

		_fan_in = other._fan_in;
		_fan_out = other._fan_out;

		_mat = other._mat;

		_initializer = other._initializer;

		_dropout = other._dropout;

		if (other._act) {
			_act = other._act->copy();
			_dact = _act->derivative();
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
	if (_act)
		delete _act;

	if (_dact)
		delete _dact;
}

// Getters and setters
template <class T>
size_t Layer <T> ::get_fan_in() const
{
	return _fan_in;
}

template <class T>
size_t Layer <T> ::get_fan_out() const
{
	return _fan_out;
}

template <class T>
void Layer <T> ::set_fan_in(size_t fan_in)
{
	_fan_in = fan_in;
	
	if (_fan_in * _fan_out > 0)
		_mat = Matrix <T> (_fan_out, _fan_in + 1);
}

template <class T>
void Layer <T> ::enable_dropout() const
{
	_dp_enable = true;
}

template <class T>
void Layer <T> ::disable_dropout() const
{
	_dp_enable = false;
}

#ifndef __AVR	// Does not support AVR

// Reading and writing
template <class T>
void Layer <T> ::write(std::ofstream &fout) const
{
	size_t r = _mat.get_rows();
	size_t c = _mat.get_cols();

	fout.write((char *) &r, sizeof(size_t));
	fout.write((char *) &c, sizeof(size_t));

	_mat.write(fout);
	_act->write(fout);
}

template <class T>
void Layer <T> ::read(std::ifstream &fin)
{
	size_t r;
	size_t c;

	fin.read((char *) &r, sizeof(size_t));
	fin.read((char *) &c, sizeof(size_t));

	_mat = Matrix <T> (r, c, T(0));

	_mat.read(fin);

	_act = Activation <T> ::load(fin);
}

#endif		// Does not support AVR

// Initializer
template <class T>
void Layer <T> ::initialize()
{
	_mat.randomize(_initializer);
}

// Computation (TODO: why is there two?)
template <class T>
inline Vector <T> Layer <T> ::forward_propogate(const Vector <T> &in)
{
	if (!_dp_enable)
		return _act->compute(_mat * in.append_above(T (1)));

	Vector <T> out = _act->compute(_mat * in.append_above(T (1)));
	if (_dropout > 0)
		out.nullify(_dropout, _unit);

	return out;
}

template <class T>
inline void Layer <T> ::forward_propogate(Vector <T> &in1, Vector <T> &in2)
{
	in2 = apt_and_mult(_mat, in1);
	in1 = _act->compute(in2);

	// Apply dropout (only if necessary)
	if (_dp_enable && _dropout > 0)
		in1.nullify(_dropout, _unit);
}

template <class T>
inline void Layer <T> ::apply_gradient(const Matrix <T> &J)
{
	_mat += J;
}

template <class T>
void Layer <T> ::print() const
{

#ifdef __AVR

	Serial.print("W = ");
	Serial.println(_mat.display());

#else

	std::cout << "W = " << _mat << std::endl;

#endif

}

template <class T>
Layer <T> &Layer <T> ::operator+=(const Matrix <T> &M)
{
	_mat += M;

	return *this;
}

// Friends
template <class T>
Layer <T> operator+(const Layer <T> &L, const Matrix <T> &M)
{
	Layer <T> out = L;

	out._mat += M;

	return out;
}

template <class T>
Layer <T> operator-(const Layer <T> &L, const Matrix <T> &M)
{
	Layer <T> out = L;

	out._mat -= M;

	return out;
}

}

}

#endif
