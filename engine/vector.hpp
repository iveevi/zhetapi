#ifndef VECTOR_H_
#define VECTOR_H_

// C/C++ headers
#ifdef __AVR	// Does not support AVR

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#else

#include <cmath>
#include <functional>

#endif		// Does not support AVR

// Engine headers
#ifdef ZHP_CUDA

#include <cuda/matrix.cuh>

#else

#include <matrix.hpp>

#endif

namespace zhetapi {

// Forward declarations
template <class T>
class Vector;

#ifndef __AVR	// Does not support AVR

// Tensor_type operations
template <class T>
struct Vector_type : std::false_type {};

template <class T>
struct Vector_type <Vector <T>> : std::true_type {};

template <class T>
bool is_vector_type()
{
	return Vector_type <T> ::value;
}

#endif		// Does not support AVR

/**
 * @brief Represents a vector in mathematics, on the scalar field corresponding
 * to T. Derived from the matrix class.
 * */
template <class T>
class Vector : public Matrix <T> {
public:
	Vector();
	Vector(const Vector &);
	Vector(const Matrix <T> &);

	Vector(size_t);
	Vector(size_t, T);
	Vector(size_t, T *, bool = true);

	// Lambda constructors
	AVR_SWITCH(
		Vector(size_t, T (*)(size_t)),
		Vector(size_t, std::function <T (size_t)>)
	);

	AVR_SWITCH(
		Vector(size_t, T *(*)(size_t)),
		Vector(size_t, std::function <T *(size_t)>)
	);

	AVR_IGNORE(Vector(const std::vector <T> &);)
	AVR_IGNORE(Vector(const std::initializer_list <T> &);)
	
	// Cross-type operations
	template <class A>
	explicit Vector(const Vector <A> &);

	// Assignment
	Vector &operator=(const Vector &);
	Vector &operator=(const Matrix <T> &);

	// Indexing
	T &operator[](size_t);
	const T &operator[](size_t) const;
	
	// Properties
	size_t size() const;

	// The three major components
	T &x();
	T &y();
	T &z();
	
	const T &x() const;
	const T &y() const;
	const T &z() const;

	// Direction of the vector (radians)
	T arg() const;
	
	// Min and max value
	T min() const;
	T max() const;

	// Min and max index
	size_t imin() const;
	size_t imax() const;
	
	// Modifiers
	Vector append_above(const T &) const;
	Vector append_below(const T &);

	Vector remove_top();
	Vector remove_bottom();

	// Normalization
	void normalize();

	// Maybe a different name (is the similarity justifiable?)
	Vector normalized() const;

	// Arithmetic operators
	void operator+=(const Vector &);
	void operator-=(const Vector &);

	// Static methods
	static Vector one(size_t);
	static Vector rarg(double, double);

	// Vector operations	
	template <class F, class U>
	friend U max(F, const Vector <U> &);
	
	template <class F, class U>
	friend U min(F, const Vector <U> &);

	template <class F, class U>
	friend U argmax(F, const Vector <U> &);
	
	template <class F, class U>
	friend U argmin(F, const Vector <U> &);
	
	template <class U>
	friend Vector <U> cross(const Vector <U> &, const Vector <U> &);

	// Vector concatenation
	template <class U>
	friend Vector <U> concat(const Vector <U> &, const Vector <U> &);
	
	template <class U>
	friend U inner(const Vector <U> &, const Vector <U> &);
	
	// Heterogenous inner product (assumes first underlying type)
	template <class U, class V>
	friend U inner(const Vector <U> &, const Vector <V> &);

	// Exceptions (put in matrix later)
	class index_out_of_bounds {};
};

// Primitive operations for all systems (including embedded)
#include <primitives/vector_prims.hpp>

// Additional operations for common systems
#ifndef __AVR

#include <vector_cpu.hpp>

#endif


}

#endif
