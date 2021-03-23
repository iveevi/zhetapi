template <class T>
Matrix <T> ::Matrix()
		: __rows(0), __cols(0),
		Tensor <T> () {}

// Owner implies that the vector object will take care of the deallocation
template <class T>
__cuda_dual_prefix
Matrix <T> ::Matrix(size_t rs, size_t cs, T *arr, bool slice)
{
	this->__size = rs * cs;

	__rows = rs;
	__cols = cs;

	this->__dim = new size_t[2];

	this->__dim[0] = rs;
	this->__dim[1] = cs;

	this->__array = arr;

	this->__sliced = slice;
}

template <class T>
T Matrix <T> ::norm() const
{
	T sum = 0;

	for (size_t i = 0; i < this->__size; i++)
		sum += this->__array[i] * this->__array[i];

	return sqrt(sum);
}

template <class T>
psize_t Matrix <T> ::get_dimensions() const
{
	return {__rows, __cols};
}

template <class T>
Matrix <T> Matrix <T> ::slice(const psize_t &start, const psize_t &end) const
{
	/* The following asserts make sure the pairs
	 * are in bounds of the Matrix and that they
	 * are in order.
	 */
	assert(start.first <= end.first && start.second <= end.second);

	assert(start.first < __rows && start.second < __cols);
	assert(end.first < __rows && end.second < __cols);

	/* Slicing is inclusive of the last
	 * Vector passed.
	 */
	return Matrix <T> (
		end.first - start.first + 1,
		end.second - start.second + 1,
		[&](size_t i, size_t j) {
			return this->__array[__cols * (i + start.first) + j + start.second];
		}
	);
}

/*
template <class T>
void Matrix <T> ::set(size_t row, size_t col, T val)
{
	this->__array[row][col] = val;
}

template <class T>
const T &Matrix <T> ::get(size_t row, size_t col) const
{
	return this->__array[row][col];
} */

template <class T>
Vector <T> Matrix <T> ::get_column(size_t r) const
{
	return Vector <T> (__rows,
		[&](size_t i) {
			return this->__array[__cols * i + r];
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
	for (size_t i = 0; i < __cols; i++)
		this->__array[a][i] += k * this->__array[b][i];
}

template <class T>
void Matrix <T> ::multiply_row(size_t a, T k)
{
	for (size_t i = 0; i < __cols; i++)
		this->__array[a][i] *= k;
}

template <class T>
T Matrix <T> ::determinant() const
{
	return determinant(*this);
}

template <class T>
T Matrix <T> ::minor(const psize_t &pr) const
{
	Matrix <T> out(__rows - 1, __cols - 1);

	size_t a = 0;

	for (size_t i = 0; i < __rows; i++) {
		size_t b = 0;
		if (i == pr.first)
			continue;

		for (size_t j = 0; j < __cols; j++) {
			if (j == pr.second)
				continue;

			out[a][b++] = this->__array[i * __cols + j];
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
	return Matrix(__rows, __cols,
		[&](size_t i, size_t j) {
			return cofactor(i, j);
		}
	);
}

template <class T>
bool Matrix <T> ::symmetric() const
{
	for (size_t i = 0; i < __rows; i++) {
		for (size_t j = 0; j < __cols; j++) {
			if (this->__array[i][j] != this->__array[j][i])
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
	assert((a.__rows == a.__cols) && (a.__rows > 0));

	size_t n;
	size_t t;
	
	n = a.__rows;

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
		: __rows(other.__rows),
		__cols(other.__cols),
		Tensor <T> (other.__rows, other.__cols)
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] = other.__array[i];
}

// TODO: Do all initialization inline or use Tensor copy constructor
template <class T>
Matrix <T> ::Matrix(const Vector <T> &other)
		: __rows(other.__rows), __cols(1),
		Tensor <T> (other.__rows, 1)
{
	for (size_t i = 0; i < __rows; i++)
		this->__array[i] = other.__array[i];
}

template <class T>
Matrix <T> ::Matrix(const Matrix <T> &other, T k)
		: __rows(other.__rows),
		__cols(other.__cols)
{
	if (this != &other) {
		// Use a macro
		this->__array = new T[other.__size];
		this->__rows = other.__rows;
		this->__cols = other.__cols;

		this->__size = other.__size;
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = k * other.__array[i];

		this->__dims = 2;
		this->__dim = new size_t[2];

		this->__dim[0] = this->__rows;
		this->__dim[1] = this->__cols;
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, T val)
		: Tensor <T> (rs, cs)
{
	__rows = rs;
	__cols = cs;
	
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] = val;
}

template <class T>
const Matrix <T> &Matrix <T> ::operator=(const Matrix <T> &other)
{
	if (this != &other) {
		this->clear();
		
		__rows = other.__rows;
		__cols = other.__cols;

		this->__size = __rows * __cols;

		this->__array = new T[this->__size];
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = other.__array[i];
		
		this->__dims = 2;
		this->__dim = new size_t[2];

		this->__dim[0] = __rows;
		this->__dim[1] = __cols;
	}

	return *this;
}

template <class T>
void Matrix <T> ::resize(size_t rs, size_t cs)
{
	if (rs != __rows || cs != __cols) {
		__rows = rs;
		__cols = cs;

		this->__size = rs * cs;

		this->clear();

		this->__array = new T[this->__size];
		
		if (!this->__dim) {
			this->__dims = 2;
			this->__dim = new size_t[2];

			this->__dim[0] = this->__rows;
			this->__dim[1] = this->__cols;
		}
	}
}

template <class T>
T *Matrix <T> ::operator[](size_t i)
{
	return (this->__array + i * __cols);
}

template <class T>
const T *Matrix <T> ::operator[](size_t i) const
{
	return (this->__array + i * __cols);
}

template <class T>
size_t Matrix <T> ::get_rows() const
{
	return __rows;
}

template <class T>
size_t Matrix <T> ::get_cols() const
{
	return __cols;
}

template <class T>
Matrix <T> Matrix <T> ::transpose() const
{
	return Matrix(__cols, __rows,
		[&](size_t i, size_t j) {
			return this->__array[j * __cols + i];
		}
	);
}

template <class T>
void Matrix <T> ::row_shur(const Vector <T> &other)
{
	// Not strict (yet)
	for (size_t i = 0; i < __rows; i++) {
		T *arr = &(this->__array[i * __cols]);

		for (size_t j = 0; j < __cols; j++)
			arr[j] *= other.__array[i];
	}
}

template <class T>
void Matrix <T> ::stable_shur(const Matrix <T> &other)
{
	if (!((other.get_rows() == __rows)
		&& (other.get_cols() == __cols)))
		throw typename Matrix <T> ::dimension_mismatch();

	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] *= other.__array[i];
}

template <class T>
void Matrix <T> ::stable_shur_relaxed(const Matrix <T> &other)
{
	// Loop for the limits of the other
	for (size_t i = 0; i < other.__size; i++)
		this->__array[i] *= other.__array[i];
}



template <class T>
void Matrix <T> ::operator+=(const Matrix <T> &other)
{
	assert(__rows == other.__rows && __cols == other.__cols);

	for (size_t i = 0; i < __rows; i++) {
		for (size_t j = 0; j < __cols; j++)
			this->__array[i * __cols + j] += other.__array[i * __cols + j];
	}
}

template <class T>
void Matrix <T> ::operator-=(const Matrix <T> &other)
{
	assert(__rows == other.__rows && __cols == other.__cols);

	for (size_t i = 0; i < __rows; i++) {
		for (size_t j = 0; j < __cols; j++)
			this->__array[i * __cols + j] -= other.__array[i * __cols + j];
	}
}

// TODO: Remove as it is done in tensor already
template <class T>
void Matrix <T> ::operator*=(const T &x)
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] *= x;
}

template <class T>
void Matrix <T> ::operator/=(const T &x)
{
	for (size_t i = 0; i < this->__size; i++)
		this->__array[i] /= x;
}

template <class T>
Matrix <T> operator+(const Matrix <T> &a, const Matrix <T> &b)
{
	assert(a.__rows == b.__rows && a.__cols == b.__cols);
	return Matrix <T> (a.__rows, a.__cols,
		[&](size_t i, size_t j) {
			return a[i][j] + b[i][j];
		}
	);
}

template <class T>
Matrix <T> operator-(const Matrix <T> &a, const Matrix <T> &b)
{
	assert(a.__rows == b.__rows && a.__cols == b.__cols);
	return Matrix <T> (a.__rows, a.__cols,
		[&](size_t i, size_t j) {
			return a[i][j] - b[i][j];
		}
	);
}

/*
template <class T>
Matrix <T> operator*(const Matrix <T> &A, const Matrix <T> &B)
{
	if (A.__cols != B.__rows)
		throw typename Matrix <T> ::dimension_mismatch();
	
	size_t rs = A.__rows;
	size_t cs = B.__cols;

	size_t kmax = B.__rows;

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
	if (A.__cols != B.__rows)
                throw typename Matrix <T> ::dimension_mismatch();

        size_t rs = A.__rows;
        size_t cs = B.__cols;

        size_t kmax = B.__rows;

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
	return Matrix <T> (a.__rows, a.__cols,
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
	return Matrix <T> (a.__rows, a.__cols,
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
	if (A.__cols != B.__rows)
		throw typename Matrix <T> ::dimension_mismatch();
	
	if (A.__rows != C.__rows || B.__cols != C.__cols)
		throw typename Matrix <T> ::dimension_mismatch();
	
	size_t rs = A.__rows;
	size_t cs = B.__cols;

	size_t kmax = B.__rows;

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
	if (A.__cols != B.__rows)
		throw typename Matrix <T> ::dimension_mismatch();
	
	if (A.__rows != C.__rows || B.__cols != C.__cols)
		throw typename Matrix <T> ::dimension_mismatch();
	
	size_t rs = A.__rows;
	size_t cs = B.__cols;

	size_t kmax = B.__rows;

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

// Externally defined methods
template <class T>
Matrix <T> Tensor <T> ::cast_to_matrix(size_t r, size_t c) const
{
	// Return a slice-vector
	return Matrix <T> (r, c, __array);
}