#ifndef VECTOR_CPU_H_
#define VECTOR_CPU_H_

namespace zhetapi {

/**
 * @brief Constructs a vector out of a list of components.
 *
 * @param ref the list of components.
 */
template <class T>
Vector <T> ::Vector(const std::vector <T> &ref)
		: Matrix <T> (ref) {}

/**
 * @brief Constructs a vector out of a list of components.
 *
 * @param ref the list of components.
 */
template <class T>
Vector <T> ::Vector(const std::initializer_list <T> &ref)
		: Vector(std::vector <T> (ref)) {}

/**
 * @brief Size constructor. Each component is evaluated from a function which
 * depends on the index.
 *
 * @param rs the number of rows (size) of the vector.
 * @param gen a pointer to the function that generates the coefficients.
 */
template <class T>
Vector <T> ::Vector(size_t rs, std::function <T (size_t)> gen)
	        : Matrix <T> (rs, 1, gen) {}

/**
 * @brief Size constructor. Each component is evaluated from a function which
 * depends on the index.
 *
 * @param rs the number of rows (size) of the vector.
 * @param gen a pointer to the function that generates pointers to the
 * coefficients.
 */
template <class T>
Vector <T> ::Vector(size_t rs, std::function <T *(size_t)> gen)
	        : Matrix <T> (rs, 1, gen) {}

/**
 * @brief Heterogenous copy constructor.
 *
 * @param other the reference vector (to be copied from).
 */
template <class T>
template <class A>
Vector <T> ::Vector(const Vector <A> &other)
{
	// TODO: Add a new function for this
	// TODO: put this function into primitives
	// TODO: use member initializer list
	this->_array = new T[other.size()];
	this->_size = other.size();
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] = other[i];

	this->_dims = 1;
	this->_dim = new size_t[1];
	this->_dim[0] = this->_size;
}

template <class T>
Vector <T> Vector <T> ::operator()(std::function <T (T)> ftn)
{
	return Vector <T> (this->_size,
		[&](size_t i) {
			return ftn(this->_array[i]);
		}
	);
}

template <class T>
T Vector <T> ::sum(std::function <T (T)> ftn)
{
	T s = 0;
	for (size_t i = 0; i < this->_size; i++)
		s += ftn(this->_array[i]);

	return s;
}

template <class T>
T Vector <T> ::product(std::function <T (T)> ftn)
{
	T p = 1;
	for (size_t i = 0; i < this->_size; i++)
		p *= ftn(this->_array[i]);

	return p;
}

/**
 * @brief Returns a vector with normalized components (length of 1). The
 * direction is preserved, as with normalization.
 *
 * @return The normalized vector.
 */
template <class T>
Vector <T> Vector <T> ::normalized() const
{
	std::vector <T> out;

	T dt = this->norm();

	for (size_t i = 0; i < this->_size; i++)
		out.push_back((*this)[i]/dt);

	return Vector(out);
}

/**
 * @brief Add and assignment operator.
 *
 * TODO: Needs to return itself
 *
 * @param the vector that will be added to this.
 */
template <class T>
void Vector <T> ::operator+=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] += a._array[i];
}

/**
 * @brief Subtract and assignment operator.
 *
 * TODO: Needs to return itself
 *
 * @param the vector that will be subtracted from this.
 */
template <class T>
void Vector <T> ::operator-=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] -= a._array[i];
}

}

#endif
