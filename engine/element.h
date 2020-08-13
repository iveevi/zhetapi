#ifndef ELEMENT_H_
#define ELEMENT_H_

#include "matrix.h"

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
	element(const std::initializer_list <T> &);

	element(size_t, T *);
	element(size_t = 0, T = T());
	
	element(size_t, std::function <T (size_t)>);
	element(size_t, std::function <T *(size_t)>);

	const element <T> &operator=(const matrix <T> &);

	size_t size() const;

	T &operator[](size_t);
	const T &operator[](size_t) const;

	// Concatenating vectors
	element append_above(const element &);
	element append_above(const T &);
	
	element append_below(const element &);
	element append_below(const T &);

	T norm() const;

	void normalize();

	element normalized();

	template <class U>
	friend U inner(const element <U> &, const element <U> &);

	template <class U>
	friend element <U> cross(const element <U> &, const element <U> &);
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
element <T> ::element(const std::initializer_list <T> &ref)
	: element(std::vector <T> (ref)) {}

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
element <T> ::element(size_t rs, std::function <T *(size_t)> gen)
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
size_t element <T> ::size() const
{
	return this->rows;
}

template <class T>
T &element <T> ::operator[](size_t i)
{
	return this->m_array[i][0];
}

template <class T>
const T &element <T> ::operator[](size_t i) const
{
	return this->m_array[i][0];
}

template <class T>
element <T> element <T> ::append_above(const element <T> &v)
{
	size_t t_sz = size();
	size_t v_sz = v.size();

	std::vector <T> total;

	for (size_t i = 0; i < v_sz; i++)
		total.push_back(v[i]);

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	return element(total);
}

template <class T>
element <T> element <T> ::append_above(const T &x)
{
	size_t t_sz = size();

	std::vector <T> total {x};

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	return element(total);
}

template <class T>
element <T> element <T> ::append_below(const element <T> &v)
{
	size_t t_sz = size();
	size_t v_sz = v.size();

	std::vector <T> total;

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	for (size_t i = 0; i < v_sz; i++)
		total.push_back(v[i]);

	return element(total);
}

template <class T>
element <T> element <T> ::append_below(const T &x)
{
	size_t t_sz = size();

	std::vector <T> total;

	for (size_t i = 0; i < t_sz; i++)
		total.push_back((*this)[i]);

	total.push_back(x);

	return element(total);
}

template <class T>
T element <T> ::norm() const
{
	return sqrt(inner(*this, *this));
}

template <class T>
void element <T> ::normalize()
{
	T dt = norm();

	for (size_t i = 0; i < size(); i++)
		(*this)[i] /= dt;
}

template <class T>
element <T> element <T> ::normalized()
{
	std::vector <T> out;

	T dt = norm();

	for (size_t i = 0; i < size(); i++)
		out.push_back((*this)[i]/dt);

	return element(out);
}

template <class T>
T inner(const element <T> &a, const element <T> &b)
{
	T acc = 0;

	assert(a.size() == b.size());
	for (size_t i = 0; i < a.rows; i++)
		acc += a[i] * b[i];

	return acc;
}

template <class T>
T cross(const element <T> &a, const element <T> &b)
{
	assert(a.size() == b.size() == 3);

	return {
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]
	};
}

#endif
