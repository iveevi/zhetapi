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

template <class T>
Vector <T> ::Vector(const std::initializer_list <T> &ref)
		: Vector(std::vector <T> (ref)) {}

template <class T>
template <class A>
Vector <T> ::Vector(const Vector <A> &other)
{
	if (is_vector_type <A> ()) {
		// Add a new function for this
		this->_array = new T[other.size()];
		this->_rows = other.get_rows();
		this->_cols = other.get_cols();

		this->_size = other.size();
		for (size_t i = 0; i < this->_size; i++)
			this->_array[i] = other[i];
		
		this->_dims = 1;
		this->_dim = new size_t[1];

		this->_dim[0] = this->_size;
	}
}

template <class T>
Vector <T> Vector <T> ::normalized() const
{
	std::vector <T> out;

	T dt = this->norm();

	for (size_t i = 0; i < size(); i++)
		out.push_back((*this)[i]/dt);

	return Vector(out);
}

template <class T>
void Vector <T> ::operator+=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] += a._array[i];
}

template <class T>
void Vector <T> ::operator-=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] -= a._array[i];
}

template <class T>
Vector <T> ::Vector(size_t rs, std::function <T (size_t)> gen)
	        : Matrix <T> (rs, 1, gen) {}

template <class T>
Vector <T> ::Vector(size_t rs, std::function <T *(size_t)> gen)
	        : Matrix <T> (rs, 1, gen) {}

}

#endif
