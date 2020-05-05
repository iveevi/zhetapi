#ifndef ELEMENT_H_
#define ELEMENT_H_

#define DEF_D	3
#define DEF_T	double

/**
 * @brief Representative
 * of a general element,
 * with components of type
 * T. In relation to a matrix,
 * these are only column vectors:
 * transpose them to get the
 * corresponding row vectors.
 */
template <class T>
class element : public matrix <T> {
public:
	element(T *);
	element(const element &);
	element(const matrix <T> &);

	element(const std::vector <T> &);
	element(size_t, T *);
	element(size_t = 0, T = T());
	element(size_t, std::function <T (size_t)>);

	const element <T> &operator=(const matrix <T> &);

	size_t length() const;

	T operator[](size_t);
	const T &operator[](size_t) const;

	T norm() const;
	const element <T> &normalize() const;

	template <class U>
	friend U inner(const element <U> &, const element <U> &);
};

template <class T>
element <T> ::element(T *ref)
{
}

template <class T>
element <T> ::element(const element &other) : matrix <T> (other) {}

template <class T>
element <T> ::element(const matrix <T> &other)
{
	*this = other;
}

template <class T>
element <T> ::element(const std::vector <T> &ref) : matrix <T> (ref)
{
	/* std::vector <std::vector <T>> pass(ref.size());
	for (auto t : ref)
		pass.push_back({t});
	*this = matrix <T> (pass); */
}

template <class T>
element <T> ::element(size_t rs, T *ref)
{

}

template <class T>
element <T> ::element(size_t rs, T def) : matrix <T> (rs, 1, def) {}

template <class T>
element <T> ::element(size_t rs, std::function <T (size_t)> gen)
	: matrix <T> (rs, 1, gen) {}

template <class T>
const element <T> &element <T> ::operator=(const matrix <T> &other)
{
	element <T> *out = new element <T> (other.get_rows(),
		[&](size_t i) {
			return other[i][0];
	});

	*this = *out;

	return *this;
}

template <class T>
size_t element <T> ::length() const
{
	return this->rows;
}

template <class T>
T element <T> ::operator[](size_t i)
{
	return this->m_array[i][0];
}

template <class T>
const T &element <T> ::operator[](size_t i) const
{
	return this->m_array[i][0];
}

template <class T>
T element <T> ::norm() const
{
	return sqrt(inner(*this, *this));
}

template <class T>
const element <T> &element <T> ::normalize() const
{
	T dt = norm();

	element <T> *out = new element <T> (this->rows,
		[&](size_t i) {
			return (*this)[i]/dt;
	});

	return *out;
}

template <class T>
T inner(const element <T> &a, const element <T> &b)
{
	T acc = 0;

	assert(a.length() == b.length());
	for (size_t i = 0; i < a.rows; i++)
		acc += a[i] * b[i];

	return acc;
}

#endif
