#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <cassert>

template <class T>
class matrix {
	T **m_array;

	size_t rows;
	size_t cols;
public:
	matrix(T **);
	matrix(const std::vector <std::vector <T>> &);

	matrix(size_t, size_t, T);
	matrix(size_t, size_t, T, T **);

	size_t get_rows() const;
	size_t get_cols() const;

	void set(size_t, size_t, T);
	const T &get(size_t, size_t) const;

	T *operator[](size_t);
	const T *operator[](size_t) const;

	const T &determinant();
	const matrix &inverse();

	std::string display() const;

	template <class U>
	friend const matrix &operator*(const matrix <U> &, const matrix <U> &);
};

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
std::string matrix <T> ::display() const
{
	std::string out;

	std::string row;
	for (int i = 0; i < rows; i++) {
		row = '|';
		
		for (int j = 0; j < cols; j++) {
			row += to_string(m_array[i][j]);
			if (j != cols - 1)
				row += "\t";
		}

		row += '|';

		out += row + "\n";
	}

	return out;
}

#endif
