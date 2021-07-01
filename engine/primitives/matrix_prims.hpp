#ifndef MATRIX_PRIMITIVES_H_
#define MATRIX_PRIMITIVES_H_

namespace zhetapi {

// Static
template <class T>
T Matrix <T> ::EPSILON = static_cast <T> (1e-10);

template <class T>
Matrix <T> ::Matrix() : Tensor <T> (), _rows(0), _cols(0) {}

#ifdef __AVR

// Lambda constructors
template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, T (*gen)(size_t))
                : Tensor <T> (rs, cs), _rows(rs), _cols(cs)
{
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[_cols * i + j] = gen(i);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, T *(*gen)(size_t))
                : Tensor <T> (rs, cs), _rows(rs), _cols(cs)
{
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[_cols * i + j] = *gen(i);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, T (*gen)(size_t, size_t))
		: Tensor <T> (rs, cs), _rows(rs), _cols(cs)
{
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[_cols * i + j] = gen(i, j);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, T *(*gen)(size_t, size_t))
		: Tensor <T> (rs, cs), _rows(rs), _cols(cs)
{
	this->_array = new T[_rows * _cols];
	for (int i = 0; i < _rows; i++) {
		for (int j = 0; j < _cols; j++)
			this->_array[_cols * i + j] = *gen(i, j);
	}
}

#endif

// Owner implies that the vector object will take care of the deallocation
template <class T>
__cuda_dual__
Matrix <T> ::Matrix(size_t rs, size_t cs, T *arr, bool slice)
{
	this->_size = rs * cs;

	_rows = rs;
	_cols = cs;

	this->_dim = new size_t[2];

	this->_dim[0] = rs;
	this->_dim[1] = cs;

	this->_array = arr;

	this->_arr_sliced = slice;
}

/**
 * @brief Component retrieval.
 *
 * @param i row index.
 * @param j column index.
 *
 * @return the component at row i and column j.
 */
template <class T>
inline T &Matrix <T> ::get(size_t i, size_t j)
{
	return this->_array[i * _cols + j];
}

/**
 * @brief Component retrieval.
 *
 * @param i row index.
 * @param j column index.
 *
 * @return the component at row i and column j.
 */
template <class T>
inline const T &Matrix <T> ::get(size_t i, size_t j) const
{
	return this->_array[i * _cols + j];
}

template <class T>
T Matrix <T> ::norm() const
{
	T sum = 0;

	for (size_t i = 0; i < this->_size; i++)
		sum += this->_array[i] * this->_array[i];

	return sqrt(sum);
}

template <class T>
psize_t Matrix <T> ::get_dimensions() const
{
	return {_rows, _cols};
}

template <class T>
Matrix <T> Matrix <T> ::slice(const psize_t &start, const psize_t &end) const
{
	/* The following asserts make sure the pairs
	 * are in bounds of the Matrix and that they
	 * are in order.
	 */
	assert(start.first <= end.first && start.second <= end.second);

	assert(start.first < _rows && start.second < _cols);
	assert(end.first < _rows && end.second < _cols);

	/* Slicing is inclusive of the last
	 * Vector passed.
	 */
	return Matrix <T> (
		end.first - start.first + 1,
		end.second - start.second + 1,
		[&](size_t i, size_t j) {
			return this->_array[_cols * (i + start.first) + j + start.second];
		}
	);
}

/*
template <class T>
void Matrix <T> ::set(size_t row, size_t col, T val)
{
	this->_array[row][col] = val;
}

template <class T>
const T &Matrix <T> ::get(size_t row, size_t col) const
{
	return this->_array[row][col];
} */

template <class T>
Vector <T> Matrix <T> ::get_column(size_t r) const
{
	return Vector <T> (_rows,
		[&](size_t i) {
			return this->_array[_cols * i + r];
		}
	);
}

template <class T>
void Matrix <T> ::operator*=(const Matrix <T> &other)
{
	(*this) = (*this) * other;
}

// R(A) = R(A) + kR(B)
template <class T>
void Matrix <T> ::add_rows(size_t a, size_t b, T k)
{
	for (size_t i = 0; i < _cols; i++)
		this->_array[a][i] += k * this->_array[b][i];
}

template <class T>
void Matrix <T> ::multiply_row(size_t a, T k)
{
	for (size_t i = 0; i < _cols; i++)
		this->_array[a][i] *= k;
}

template <class T>
T Matrix <T> ::determinant() const
{
	return determinant(*this);
}

template <class T>
T Matrix <T> ::minor(const psize_t &pr) const
{
	Matrix <T> out(_rows - 1, _cols - 1);

	size_t a = 0;

	for (size_t i = 0; i < _rows; i++) {
		size_t b = 0;
		if (i == pr.first)
			continue;

		for (size_t j = 0; j < _cols; j++) {
			if (j == pr.second)
				continue;

			out[a][b++] = this->_array[i * _cols + j];
		}

		a++;
	}

	return determinant(out);
}

template <class T>
T Matrix <T> ::minor(size_t i, size_t j) const
{
	return minor({i, j});
}

template <class T>
T Matrix <T> ::cofactor(const psize_t &pr) const
{
	return (((pr.first + pr.second) % 2) ? -1 : 1) * minor(pr);
}

template <class T>
T Matrix <T> ::cofactor(size_t i, size_t j) const
{
	return cofactor({i, j});
}

template <class T>
Matrix <T> Matrix <T> ::inverse() const
{
	return adjugate() / determinant();
}

template <class T>
Matrix <T> Matrix <T> ::adjugate() const
{
	return cofactor().transpose();
}

template <class T>
Matrix <T> Matrix <T> ::cofactor() const
{
	return Matrix(_rows, _cols,
		[&](size_t i, size_t j) {
			return cofactor(i, j);
		}
	);
}

// TODO: put these property checks in another file

/**
 * @brief Checks whether the matrix is symmetric.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is symmetric and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_symmetric(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = i + 1; j < c; j++) {
			// Avoid using abs
			T d = get(i, j) - get(j, i);
			if (d > epsilon || d < -epsilon)
				return false;
		}
	}

	return true;
}

/**
 * @brief Checks whether the matrix is diagonal.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is diagonal and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_diagonal(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if (i == j)
				continue;

			if (get(i, j) < -epsilon || get(i, j) > epsilon)
				return false;
		}
	}

	return true;
}

/**
 * @brief Checks whether the matrix is identity matrix (for any dimension).
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is the identity and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_identity(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	if (r != c)
		return false;

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if (i == j) {
				T d = 1 - get(i, j);
				if (d < -epsilon || d > epsilon)
					return false;
			} else if (get(i, j) < -epsilon || get(i, j) > epsilon) {
				return false;
			}
		}
	}

	return true;
}

/**
 * @brief Checks whether the matrix is orthogonal.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is orthogonal and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_orthogonal(const T &epsilon) const
{
	return is_identity(*this * transpose());
}

/**
 * @brief Checks whether the matrix is lower triangular.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is lower triangular and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_lower_triangular(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if ((i > j) && get(i, j) > epsilon)
				return false;
		}
	}

	return true;
}

/**
 * @brief Checks whether the matrix is upper triangular.
 *
 * @param epsilon the maximum tolerance. Defaults to EPSILON.
 *
 * @return \c true if the matrix is upper triangular and \c false otherwise.
 */
template <class T>
bool Matrix <T> ::is_upper_triangular(const T &epsilon) const
{
	size_t r = get_rows();
	size_t c = get_cols();

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			if ((i < j) && get(i, j) > epsilon)
				return false;
		}
	}

	return true;
}

template <class T>
Matrix <T> Matrix <T> ::identity(size_t dim)
{
	return Matrix(dim, dim,
		[](size_t i, size_t j) {
			return (i == j) ? 1 : 0;
		}
	);
}

// Private helper methods
template <class T>
T Matrix <T> ::determinant(const Matrix <T> &a) const
{
	/* The determinant of an abitrary
	 * Matrix is defined only if it
	 * is a square Matrix.
	 */
	assert((a._rows == a._cols) && (a._rows > 0));

	size_t n;
	size_t t;

	n = a._rows;

	if (n == 1)
		return a[0][0];
	if (n == 2)
		return a[0][0] * a[1][1] - a[1][0] * a[0][1];

	T det = 0;

	for (size_t i = 0; i < n; i++) {
		Matrix <T> temp(n - 1, n - 1);

		for (size_t j = 0; j < n - 1; j++) {
			t = 0;

			for (size_t k = 0; k < n; k++) {
				if (k == i)
					continue;
				temp[j][t++] = a[j + 1][k];
			}
		}

		det += ((i % 2) ? -1 : 1) * a[0][i] * determinant(temp);
	}

	return det;
}

template <class T>
Matrix <T> ::Matrix(const Matrix <T> &other)
		: Tensor <T> (other._rows, other._cols),
		_rows(other._rows), _cols(other._cols)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] = other._array[i];
}

// TODO: Do all initialization inline or use Tensor copy constructor
template <class T>
Matrix <T> ::Matrix(const Vector <T> &other)
		: _rows(other._rows), _cols(1),
		Tensor <T> (other._rows, 1)
{
	for (size_t i = 0; i < _rows; i++)
		this->_array[i] = other._array[i];
}

template <class T>
Matrix <T> ::Matrix(const Matrix <T> &other, T k)
		: _rows(other._rows),
		_cols(other._cols)
{
	if (this != &other) {
		// Use a macro
		this->_array = new T[other._size];
		this->_rows = other._rows;
		this->_cols = other._cols;

		this->_size = other._size;
		for (size_t i = 0; i < this->_size; i++)
			this->_array[i] = k * other._array[i];

		this->_dims = 2;
		this->_dim = new size_t[2];

		this->_dim[0] = this->_rows;
		this->_dim[1] = this->_cols;
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, T val)
		: Tensor <T> (rs, cs)
{
	_rows = rs;
	_cols = cs;

	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] = val;
}

template <class T>
const Matrix <T> &Matrix <T> ::operator=(const Matrix <T> &other)
{
	if (this != &other) {
		this->clear();

		_rows = other._rows;
		_cols = other._cols;

		this->_size = _rows * _cols;

		this->_array = new T[this->_size];
		for (size_t i = 0; i < this->_size; i++)
			this->_array[i] = other._array[i];

		this->_dims = 2;
		this->_dim = new size_t[2];

		this->_dim[0] = _rows;
		this->_dim[1] = _cols;
	}

	return *this;
}

template <class T>
void Matrix <T> ::resize(size_t rs, size_t cs)
{
	if (rs != _rows || cs != _cols) {
		_rows = rs;
		_cols = cs;

		this->_size = rs * cs;

		this->clear();

		this->_array = new T[this->_size];

		if (!this->_dim) {
			this->_dims = 2;
			this->_dim = new size_t[2];

			this->_dim[0] = this->_rows;
			this->_dim[1] = this->_cols;
		}
	}
}

template <class T>
T *Matrix <T> ::operator[](size_t i)
{
	return (this->_array + i * _cols);
}

template <class T>
const T *Matrix <T> ::operator[](size_t i) const
{
	return (this->_array + i * _cols);
}

template <class T>
inline size_t Matrix <T> ::get_rows() const
{
	return this->_dim[0];
}

template <class T>
inline size_t Matrix <T> ::get_cols() const
{
	return this->_dim[1];
}

template <class T>
Matrix <T> Matrix <T> ::transpose() const
{
	return Matrix(_cols, _rows,
		[&](size_t i, size_t j) {
			return this->_array[j * _cols + i];
		}
	);
}

template <class T>
void Matrix <T> ::row_shur(const Vector <T> &other)
{
	// Not strict (yet)
	for (size_t i = 0; i < _rows; i++) {
		T *arr = &(this->_array[i * _cols]);

		for (size_t j = 0; j < _cols; j++)
			arr[j] *= other._array[i];
	}
}

template <class T>
void Matrix <T> ::stable_shur(const Matrix <T> &other)
{
	if (!((other.get_rows() == _rows)
		&& (other.get_cols() == _cols)))
		throw typename Matrix <T> ::dimension_mismatch();

	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] *= other._array[i];
}

template <class T>
void Matrix <T> ::stable_shur_relaxed(const Matrix <T> &other)
{
	// Loop for the limits of the other
	for (size_t i = 0; i < other._size; i++)
		this->_array[i] *= other._array[i];
}



template <class T>
void Matrix <T> ::operator+=(const Matrix <T> &other)
{
	assert(_rows == other._rows && _cols == other._cols);

	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[i * _cols + j] += other._array[i * _cols + j];
	}
}

template <class T>
void Matrix <T> ::operator-=(const Matrix <T> &other)
{
	assert(_rows == other._rows && _cols == other._cols);

	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[i * _cols + j] -= other._array[i * _cols + j];
	}
}

// TODO: Remove as it is done in tensor already
template <class T>
void Matrix <T> ::operator*=(const T &x)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] *= x;
}

template <class T>
void Matrix <T> ::operator/=(const T &x)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] /= x;
}

template <class T>
Matrix <T> operator+(const Matrix <T> &a, const Matrix <T> &b)
{
	assert(a._rows == b._rows && a._cols == b._cols);
	return Matrix <T> (a._rows, a._cols,
		[&](size_t i, size_t j) {
			return a[i][j] + b[i][j];
		}
	);
}

template <class T>
Matrix <T> operator-(const Matrix <T> &a, const Matrix <T> &b)
{
	assert(a._rows == b._rows && a._cols == b._cols);
	return Matrix <T> (a._rows, a._cols,
		[&](size_t i, size_t j) {
			return a[i][j] - b[i][j];
		}
	);
}

/*
template <class T>
Matrix <T> operator*(const Matrix <T> &A, const Matrix <T> &B)
{
	if (A._cols != B._rows)
		throw typename Matrix <T> ::dimension_mismatch();

	size_t rs = A._rows;
	size_t cs = B._cols;

	size_t kmax = B._rows;

	inline_init_mat(C, rs, cs);

	for (size_t i = 0; i < rs; i++) {
		const T *Ar = A[i];
		T *Cr = C[i];

		for (size_t k = 0; k < kmax; k++) {
			const T *Br = B[k];

			T a = Ar[k];
			for (size_t j = 0; j < cs; j++)
				Cr[j] += a * Br[j];
		}
	}

	return C;
} */

template <class T, class U>
Matrix <T> operator*(const Matrix <T> &A, const Matrix <U> &B)
{
	AVR_IGNORE(
		if (A._cols != B._rows)
			throw typename Matrix <T> ::dimension_mismatch()
	);

        size_t rs = A._rows;
        size_t cs = B._cols;

        size_t kmax = B._rows;

        inline_init_mat(C, rs, cs);

        for (size_t i = 0; i < rs; i++) {
                const T *Ar = A[i];
                T *Cr = C[i];

                for (size_t k = 0; k < kmax; k++) {
                        const U *Br = B[k];

                        T a = Ar[k];
                        for (size_t j = 0; j < cs; j++)
                                Cr[j] += T(a * Br[j]);
                }
        }

	return C;
}

template <class T>
Matrix <T> operator*(const Matrix <T> &a, const T &scalar)
{
	return Matrix <T> (a._rows, a._cols,
		[&](size_t i, size_t j) {
			return a[i][j] * scalar;
		}
	);
}

template <class T>
Matrix <T> operator*(const T &scalar, const Matrix <T> &a)
{
	return a * scalar;
}

template <class T>
Matrix <T> operator/(const Matrix <T> &a, const T &scalar)
{
	return Matrix <T> (a._rows, a._cols,
		[&](size_t i, size_t j) {
			return a[i][j] / scalar;
		}
	);
}

template <class T>
Matrix <T> operator/(const T &scalar, const Matrix <T> &a)
{
	return a / scalar;
}

template <class T>
bool operator==(const Matrix <T> &a, const Matrix <T> &b)
{
	if (a.get_dimensions() != b.get_dimensions())
		return false;

	for (size_t i = 0; i  < a.get_rows(); i++) {
		for (size_t j = 0; j < a.get_cols(); j++) {
			if (a[i][j] != b[i][j])
				return false;
		}
	}

	return true;
}

template <class T>
Matrix <T> shur(const Matrix <T> &a, const Matrix <T> &b)
{
	if (!((a.get_rows() == b.get_rows())
		&& (a.get_cols() == b.get_cols())))
		throw typename Matrix <T> ::dimension_mismatch();

	return Matrix <T> (a.get_rows(), b.get_cols(),
		[&](size_t i, size_t j) {
			return a[i][j] * b[i][j];
		}
	);
}

template <class T>
Matrix <T> inv_shur(const Matrix <T> &a, const Matrix <T> &b)
{
	if (!((a.get_rows() == b.get_rows())
		&& (a.get_cols() == b.get_cols())))
		throw typename Matrix <T> ::dimension_mismatch();

	return Matrix <T> (a.get_rows(), b.get_cols(),
		[&](size_t i, size_t j) {
			return a[i][j] / b[i][j];
		}
	);
}

// Computes A * B + C
template <class T, class U, class V>
Matrix <T> fma(const Matrix <T> &A, const Matrix <U> &B, const Matrix <V> &C)
{
	if (A._cols != B._rows)
		throw typename Matrix <T> ::dimension_mismatch();

	if (A._rows != C._rows || B._cols != C._cols)
		throw typename Matrix <T> ::dimension_mismatch();

	size_t rs = A._rows;
	size_t cs = B._cols;

	size_t kmax = B._rows;

	Matrix <T> D = C;

	for (size_t i = 0; i < rs; i++) {
		const T *Ar = A[i];
		T *Dr = D[i];

		for (size_t k = 0; k < kmax; k++) {
			const U *Br = B[k];

			T a = Ar[k];
			for (size_t j = 0; j < cs; j++)
				Dr[j] += T(a * Br[j]);
		}
	}

	return D;
}

// Computes ka * A * B + kb * C
template <class T, class U, class V>
Matrix <T> fmak(const Matrix <T> &A, const Matrix <U> &B, const Matrix <V> &C, T ka, T kb)
{
	if (A._cols != B._rows)
		throw typename Matrix <T> ::dimension_mismatch();

	if (A._rows != C._rows || B._cols != C._cols)
		throw typename Matrix <T> ::dimension_mismatch();

	size_t rs = A._rows;
	size_t cs = B._cols;

	size_t kmax = B._rows;

	Matrix <T> D(C, kb);

	for (size_t i = 0; i < rs; i++) {
		const T *Ar = A[i];
		T *Dr = D[i];

		for (size_t k = 0; k < kmax; k++) {
			const U *Br = B[k];

			T a = Ar[k] * ka;
			for (size_t j = 0; j < cs; j++)
				Dr[j] += T(a * Br[j]);
		}
	}

	return D;
}

#ifdef __AVR

template <class T>
String Matrix <T> ::display() const
{
	String out = "[";

	for (size_t i = 0; i < _rows; i++) {
		if (_cols > 1) {
			out += '[';

			for (size_t j = 0; j < _cols; j++) {
				out += String(this->_array[i * _cols + j]);

				if (j != _cols - 1)
					out += ", ";
			}

			out += ']';
		} else {
			out += String(this->_array[i * _cols]);
		}

		if (i < _rows - 1)
			out += ", ";
	}

	return out + "]";
}

#endif

// Externally defined methods
template <class T>
Matrix <T> Tensor <T> ::cast_to_matrix(size_t r, size_t c) const
{
	// Return a slice-vector
	return Matrix <T> (r, c, _array);
}

}

#endif
