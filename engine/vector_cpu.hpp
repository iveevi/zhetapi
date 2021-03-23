#ifndef VECTOR_CPU_H_
#define VECTOR_CPU_H_

template <class T>
Vector <T> ::Vector() : Matrix <T> () {}

template <class T>
Vector <T> ::Vector(const Vector &other) : Matrix <T> (other.size(), 1, T())
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] = other.__array[i];
}

template <class T>
Vector <T> ::Vector(const Matrix <T> &other) : Matrix <T> (other.get_rows(), 1, T())
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] = other[0][i];
}

template <class T>
Vector <T> ::Vector(size_t rs, T def) : Matrix <T> (rs, 1, def) {}

// FIXME: Delegate Matrix constructor
template <class T>
Vector <T> ::Vector(size_t rs, T *ref, bool slice) : Matrix <T> (rs, 1, ref, slice) {}

template <class T>
Vector <T> &Vector <T> ::operator=(const Vector <T> &other)
{
	if (this != &other) {
		this->clear();

		this->__array = new T[other.__size];
		this->__rows = other.__rows;
		this->__cols = other.__cols;

		this->__size = other.__size;
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = other.__array[i];
		
		this->__dims = 1;
		this->__dim = new size_t[1];

		this->__dim[0] = this->__size;
	}

	return *this;
}

template <class T>
Vector <T> &Vector <T> ::operator=(const Matrix <T> &other)
{
	if (this != &other) {
		*this = Vector(other.get_rows(), T());

		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = other[0][i];
	}

	return *this;
}

template <class T>
T &Vector <T> ::operator[](size_t i)
{
	return this->__array[i];
}

template <class T>
const T &Vector <T> ::operator[](size_t i) const
{
	return this->__array[i];
}

template <class T>
size_t Vector <T> ::size() const
{
	return this->__size;
}

template <class T>
void Vector <T> ::operator+=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] += a.__array[i];
}

template <class T>
void Vector <T> ::operator-=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] -= a.__array[i];
}

#ifndef __AVR   // AVR support

template <class T>
Vector <T> ::Vector(size_t rs, std::function <T (size_t)> gen)
	        : Matrix <T> (rs, 1, gen) {}

template <class T>
Vector <T> ::Vector(size_t rs, std::function <T *(size_t)> gen)
	        : Matrix <T> (rs, 1, gen) {}

// TODO: bring back these operations for AVR
template <class T>
Vector <T> Vector <T> ::append_above(const T &x) const
{
	size_t t_sz = size();

	std::vector <T> total {x};
	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	return Vector(total);
}

template <class T>
Vector <T> Vector <T> ::append_below(const T &x)
{
	size_t t_sz = size();

	std::vector <T> total;

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	total.push_back(x);

	return Vector(total);
}

template <class T>
Vector <T> Vector <T> ::remove_top()
{
	size_t t_sz = size();

	std::vector <T> total;
	for (size_t i = 1; i < t_sz; i++)
		total.push_back((*this)[i]);

	return Vector(total);
}

template <class T>
Vector <T> Vector <T> ::remove_bottom()
{
	size_t t_sz = size();

	std::vector <T> total;
	for (size_t i = 0; i < t_sz - 1; i++)
		total.push_back((*this)[i]);

	return Vector(total);
}

#endif          // AVR support

#endif
