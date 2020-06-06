#ifndef MATRIX_H_
#define MATRIX_H_

#include <cassert>
#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <sstream>

#include "functor.h"

#ifdef minor

#undef minor

#endif

/**
 * @brief A general matrix class
 * (could be a single row/col vector)
 * that supports conventional operations
 * that matrices in mathematics do.
 */
template <class T>
class matrix {
protected:
	T **m_array;

	size_t rows;
	size_t cols;
public:
	matrix();
	matrix(const matrix <T> &);

	matrix(T **);
	matrix(const std::vector <T> &);
	matrix(const std::vector <std::vector <T>> &);

	matrix(size_t, size_t, T = T());
	matrix(size_t, size_t, T, T **);

	matrix(size_t, size_t, std::function <T (size_t)>);
	matrix(size_t, size_t, std::function <T (size_t, size_t)>);

	~matrix();

	std::pair <size_t, size_t> get_dimensions() const;

	size_t get_rows() const;
	size_t get_cols() const;

	const matrix &slice(const std::pair <size_t, size_t> &,
			const std::pair <size_t, size_t> &) const;

	void set(size_t, size_t, T);
	const T &get(size_t, size_t) const;

	T *operator[](size_t);
	const T *operator[](size_t) const;

	void operator+=(const matrix &);
	void operator-=(const matrix &);
	void operator*=(const matrix &);
	void operator/=(const matrix &);

	T determinant() const;

	T minor(size_t, size_t) const;
	T minor(const std::pair <size_t, size_t> &) const;

	T cofactor(size_t, size_t) const;
	T cofactor(const std::pair <size_t, size_t> &) const;

	const matrix &inverse() const;
	const matrix &adjugate() const;
	const matrix &cofactor() const;
	const matrix &transpose() const;

	std::string display() const;

	template <class U>
	friend const matrix <U> &operator+(const matrix <U> &, const matrix <U> &);
	
	template <class U>
	friend const matrix <U> &operator-(const matrix <U> &, const matrix <U> &);

	template <class U>
	friend const matrix <U> &operator*(const matrix <U> &, const matrix <U> &);

	/* template <class U>
	friend const U &operator*(const matrix <U> &, const matrix <U> &); */

	template <class U>
	friend const matrix <U> &operator*(const matrix <U> &, const U &);
	
	template <class U>
	friend const matrix <U> &operator*(const U &, const matrix <U> &);
	
	template <class U>
	friend const matrix <U> &operator/(const matrix <U> &, const U &);
	
	template <class U>
	friend const matrix <U> &operator/(const U &, const matrix <U> &);

	template <class U>
	friend std::ostream &operator<<(std::ostream &, const matrix <U> &);
protected:
	T determinant(const matrix &) const;
};

template <class T>
matrix <T> ::matrix() : rows(0), cols(0), m_array(nullptr) {}

template <class T>
matrix <T> ::matrix(const matrix <T> &other) : rows(other.rows),
		cols(other.cols)
{
	m_array = new T *[rows];

	for (int i = 0; i < rows; i++) {
		m_array[i] = new T[cols];

		for (int j = 0; j < cols; j++)
			m_array[i][j] = other[i][j];
	}
}

template <class T>
matrix <T> ::matrix(T **ref)
{
	rows = sizeof(ref)/sizeof(T);

	assert(rows > 0);

	cols = sizeof(ref[0])/sizeof(T);

	assert(cols > 0);

	m_array = new T *[rows];

	for (int i = 0; i < rows; i++) {
		m_array[i] = new T[cols];

		for (int j = 0; j < cols; j++)
			m_array[i][j] = ref[i][j];
	}
}

template <class T>
matrix <T> ::matrix(const std::vector <T> &ref)
{
	rows = ref.size();

	assert(rows > 0);

	cols = 1;

	assert(cols > 0);
	
	m_array = new T *[rows];

	for (int i = 0; i < rows; i++)
		m_array[i] = new T {ref[i]};
}

template <class T>
matrix <T> ::matrix(const std::vector <std::vector <T>> &ref)
{
	rows = ref.size();

	assert(rows > 0);

	cols = ref[0].size();

	assert(cols > 0);
	
	m_array = new T *[rows];

	for (int i = 0; i < rows; i++) {
		m_array[i] = new T[cols];

		for (int j = 0; j < cols; j++) {
			assert(i < rows && j < ref[i].size());
			m_array[i][j] = ref[i][j];
		}
	}
}

template <class T>
matrix <T> ::matrix(size_t rs, size_t cs, T val)
{
	rows = rs;
	cols = cs;

	m_array = new T *[rows];

	for (int i = 0; i < rows; i++) {
		m_array[i] = new T[cols];
		
		for (int j = 0; j < cols; j++)
			m_array[i][j] = val;
	}
}

template <class T>
matrix <T> ::matrix(size_t rows, size_t cols, T val, T **ref)
{
}

template <class T>
matrix <T> ::matrix(size_t rs, size_t cs,
		std::function <T (size_t)> gen)
{
	rows = rs;
	cols = cs;

	m_array = new T *[rows];

	for (int i = 0; i < rows; i++) {
		m_array[i] = new T[cols];
		
		for (int j = 0; j < cols; j++)
			m_array[i][j] = gen(i);
	}
}

template <class T>
matrix <T> ::matrix(size_t rs, size_t cs,
		std::function <T (size_t, size_t)> gen)
{
	rows = rs;
	cols = cs;

	m_array = new T *[rows];

	for (int i = 0; i < rows; i++) {
		m_array[i] = new T[cols];
		
		for (int j = 0; j < cols; j++)
			m_array[i][j] = gen(i, j);
	}
}

template <class T>
matrix <T> ::~matrix()
{
	// delete[] m_array;
}

template <class T>
std::pair <size_t, size_t> matrix <T> ::get_dimensions() const
{
	return {rows, cols};
}

template <class T>
size_t matrix <T> ::get_rows() const
{
	return rows;
}

template <class T>
size_t matrix <T> ::get_cols() const
{
	return cols;
}

template <class T>
const matrix <T> &matrix <T> ::slice(const std::pair <size_t, size_t> &start,
		const std::pair <size_t, size_t> &end) const
{
	/* The following asserts make sure the pairs
	 * are in bounds of the matrix and that they
	 * are in order */
	assert(start.first <= end.first && start.second <= end.second);

	assert(start.first >= 0 && start.second >= 0 &&
			start.first < rows && start.second < cols);
	assert(end.first >= 0 && end.second >= 0 &&
			end.first < rows && end.second < cols);

	/* Slicing is inclusive of the last
	 * element passed */
	matrix <T> *out = new matrix <T> (end.first - start.first + 1,
			end.second - start.second + 1, [&](size_t i, size_t j) {
			return m_array[i + start.first][j + start.second];
	});

	return *out;
}

template <class T>
void matrix <T> ::set(size_t row, size_t col, T val)
{
	m_array[row][col] = val;
}

template <class T>
const T &matrix <T> ::get(size_t row, size_t col) const
{
	return m_array[row][col];
}

template <class T>
T *matrix <T> ::operator[](size_t i)
{
	return m_array[i];
}

template <class T>
const T *matrix <T> ::operator[](size_t i) const
{
	return m_array[i];
}

template <class T>
void matrix <T> ::operator+=(const matrix <T> &other)
{
	assert(rows == other.rows && cols == other.cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++)
			m_array[i][j] += other.m_array[i][j];
	}
}

template <class T>
void matrix <T> ::operator-=(const matrix <T> &other)
{
	assert(rows == other.rows && cols == other.cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++)
			m_array[i][j] -= other.m_array[i][j];
	}
}

template <class T>
void matrix <T> ::operator*=(const matrix <T> &other)
{
	(*this) = (*this) * other;
}

template <class T>
T matrix <T> ::determinant() const
{
	return determinant(*this);
}

template <class T>
T matrix <T> ::minor(const std::pair <size_t, size_t> &pr) const
{
	matrix <T> *out = new matrix <T> (rows - 1, cols - 1);

	size_t a;
	size_t b;

	a = 0;
	for (size_t i = 0; i < rows; i++) {
		b = 0;
		if (i == pr.first)
			continue;

		for (size_t j = 0; j < cols; j++) {
			if (j == pr.second)
				continue;

			(*out)[a][b++] = m_array[i][j];
		}

		a++;
	}

	T min = determinant(*out);

	delete out;

	return min;
}

template <class T>
T matrix <T> ::minor(size_t i, size_t j) const
{
	return minor({i, j});
}

template <class T>
T matrix <T> ::cofactor(const std::pair <size_t, size_t> &pr) const
{
	return (((pr.first + pr.second) % 2) ? -1 : 1) * minor(pr);
}

template <class T>
T matrix <T> ::cofactor(size_t i, size_t j) const
{
	return cofactor({i, j});
}

template <class T>
const matrix <T> &matrix <T> ::inverse() const
{
	return adjugate() / determinant();
}

template <class T>
const matrix <T> &matrix <T> ::adjugate() const
{
	return cofactor().transpose();
}

template <class T>
const matrix <T> &matrix <T> ::cofactor() const
{
	matrix <T> *out = new matrix(rows, cols, [&](size_t i, size_t j) {
		return cofactor(i, j);
	});

	return *out;
}

template <class T>
const matrix <T> &matrix <T> ::transpose() const
{
	matrix <T> *out = new matrix <T> (cols, rows, [&](size_t i, size_t j) {
		return m_array[j][i];
	});

	return *out;
}

template <class T>
std::string matrix <T> ::display() const
{
	std::ostringstream oss;
	for (int i = 0; i < rows; i++) {
		oss << '|';

		for (int j = 0; j < cols; j++) {
			oss << m_array[i][j];
			if (j != cols - 1)
				oss << "\t";
		}

		oss << '|';

		if (i < rows - 1)
			oss << "\n";
	}

	return oss.str();
}

template <class T>
const matrix <T> &operator+(const matrix <T> &a, const matrix <T> &b)
{
	assert(a.rows == b.rows && a.cols == b.cols);
	
	matrix <T> *out = new matrix <T> (a.rows, a.cols, [&](size_t i, size_t j) {
		return a[i][j] + b[i][j];
	});

	return *out;
}

template <class T>
const matrix <T> &operator-(const matrix <T> &a, const matrix <T> &b)
{
	assert(a.rows == b.rows && a.cols == b.cols);

	matrix <T> *out = new matrix <T> (a.rows, a.cols, [&](size_t i, size_t j) {
		return a[i][j] - b[i][j];
	});

	return *out;
}

template <class T>
const matrix <T> &operator*(const matrix <T> &a, const matrix <T> &b)
{
	assert(a.cols == b.rows);

	matrix <T> *out = new matrix <T> (a.rows, b.cols, [&](size_t i, size_t j) {
		T acc = 0;

		for (size_t k = 0; k < a.cols; k++) {
			acc += a[i][k] * b[k][j];
		}

		return acc;
	});

	return *out;
}

/* template <class T>
const T &matrix <T> ::operator*(const matrix <T> &a, const matrix <T> &b)
{
	assert(a.cols == b.rows && a.rows == b.cols == 1);
	return (a * b)[0][0];
} */

template <class T>
const matrix <T> &operator*(const matrix <T> &a, const T &scalar)
{
	matrix <T> *out = new matrix <T> (a.rows, a.cols, [&](size_t i, size_t j) {
		return a[i][j] * scalar;
	});

	return *out;
}

template <class T>
const matrix <T> &operator*(const T &scalar, const matrix <T> &a)
{
	return a * scalar;
}

template <class T>
const matrix <T> &operator/(const matrix <T> &a, const T &scalar)
{
	matrix <T> *out = new matrix <T> (a.rows, a.cols, [&](size_t i, size_t j) {
		return a[i][j] / scalar;
	});

	return *out;
}

template <class T>
const matrix <T> &operator/(const T &scalar, const matrix <T> &a)
{
	return a / scalar;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const matrix <T> &a)
{
	os << a.display();
	return os;
}

// Private helper methods
template <class T>
T matrix <T> ::determinant(const matrix <T> &a) const
{
	/* The determinant of an abitrary
	 * matrix is defined only if it
	 * is a square matrix */
	assert(a.rows == a.cols && a.rows > 0);

	size_t n;
	size_t t;
	
	n = a.rows;

	if (n == 1)
		return a[0][0];
	if (n == 2)
		return a[0][0] * a[1][1] - a[1][0] * a[0][1];

	T det = 0;

	matrix <T> *temp;
	for (size_t i = 0; i < n; i++) {
		temp = new matrix <T> (n - 1, n - 1);

		for (size_t j = 0; j < n - 1; j++) {
			t = 0;

			for (size_t k = 0; k < n; k++) {
				if (k == i)
					continue;
				(*temp)[j][t++] = a[j + 1][k];
			}
		}

		det += ((i % 2) ? -1 : 1) * a[0][i] * determinant(*temp);

		delete temp;
	}

	return det;
}

#endif
