#ifndef MATRIX_CPU_H_
#define MATRIX_CPU_H_

namespace zhetapi {

template <class T>
Matrix <T> ::Matrix(const std::vector <Vector <T>> &columns)
{
	size_t r = columns.size();

	if (r > 0) {
		size_t c = columns[0].size();

		this->_size = r * c;
		this->_dims = 2;

		this->_dim = new size_t[2];
		this->_dim[0] = r;
		this->_dim[1] = c;

		this->_array = new T[this->_size];

		for (size_t i = 0; i < get_rows(); i++) {
			for (size_t j = 0; j < get_cols(); j++)
				this->_array[get_cols() * i + j] = columns[j][i];
		}
	}
}

template <class T>
Matrix <T> ::Matrix(const std::initializer_list <Vector <T>> &columns)
		: Matrix(std::vector <Vector <T>> (columns)) {}

template <class T>
Matrix <T> ::Matrix(const std::vector <T> &ref)
		: Tensor <T> ({ref.size(), 1}, T())
{
	assert(get_rows() > 0);
	for (size_t i = 0; i < get_rows(); i++)
		this->_array[i] = ref[i];
}

template <class T>
Matrix <T> ::Matrix(const std::vector <std::vector <T>> &ref)
		: Tensor <T> (ref.size(), ref[0].size())
{
	assert(get_rows() > 0);
	assert(get_cols() > 0);

	for (int i = 0; i < get_rows(); i++) {
		for (int j = 0; j < get_cols(); j++) {
			assert(i < get_rows() && j < ref[i].size());

			this->_array[get_cols() * i + j] = ref[i][j];
		}
	}
}

template <class T>
Matrix <T> ::Matrix(const std::initializer_list <std::initializer_list <T>> &sq)
                : Tensor <T> (sq.size(), sq.begin()->size())
{
	assert(get_rows() > 0);
	assert(get_cols() > 0);

	size_t i = 0;
	for (auto lt : sq) {

		size_t j = 0;
		for (auto t : lt) {
			assert(i < get_rows() && j < lt.size());

			this->_array[get_cols() * i + (j++)] = t;
		}

		i++;
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T (size_t)> gen)
                : Tensor <T> (rs, cs)
{
	for (size_t i = 0; i < get_rows(); i++) {
		for (size_t j = 0; j < get_cols(); j++)
			this->_array[get_cols() * i + j] = gen(i);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T *(size_t)> gen)
                : Tensor <T> (rs, cs)
{
	for (size_t i = 0; i < get_rows(); i++) {
		for (size_t j = 0; j < get_cols(); j++)
			this->_array[get_cols() * i + j] = *gen(i);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T (size_t, size_t)> gen)
		: Tensor <T> (rs, cs)
{
	for (size_t i = 0; i < get_rows(); i++) {
		for (size_t j = 0; j < get_cols(); j++)
			this->_array[get_cols() * i + j] = gen(i, j);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T *(size_t, size_t)> gen)
		: Tensor <T> (rs, cs)
{
	get_rows() = rs;
	get_cols() = cs;

	this->_array = new T[get_rows() * get_cols()];
	for (int i = 0; i < get_rows(); i++) {
		for (int j = 0; j < get_cols(); j++)
			this->_array[get_cols() * i + j] = *gen(i, j);
	}
}

template <class T>
void Matrix <T> ::write(std::ofstream &fout) const
{
	for (size_t i = 0; i < this->_size; i++)
		fout.write((char *) &(this->_array[i]), sizeof(T));
}

template <class T>
void Matrix <T> ::read(std::ifstream &fin)
{
	for (size_t i = 0; i < this->_size; i++)
		fin.read((char *) &(this->_array[i]), sizeof(T));
}

template <class T>
void Matrix <T> ::randomize(std::function <T ()> ftr)
{
	for (size_t i = 0; i < get_rows(); i++) {
		for (size_t j = 0; j < get_cols(); j++)
			this->_array[i * get_cols() + j] = ftr();
	}
}

template <class T>
Matrix <T> Matrix <T> ::append_above(const Matrix &m)
{
	assert(get_cols() == m.get_cols());

	size_t trows = get_rows();
	size_t mrows = m.get_rows();

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < mrows; i++) {
		total.clear();

		for (size_t j = 0; j < get_cols(); j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < trows; i++) {
		total.clear();

		for (size_t j = 0; j < get_cols(); j++)
			total.push_back(this->_array[i][j]);

		row.push_back(total);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_below(const Matrix &m)
{
	assert(get_cols() == m.get_cols());

	size_t trows = get_rows();
	size_t mrows = m.get_rows();

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < trows; i++) {
		total.clear();

		for (size_t j = 0; j < get_cols(); j++)
			total.push_back(this->_array[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < mrows; i++) {
		total.clear();

		for (size_t j = 0; j < get_cols(); j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_left(const Matrix &m)
{
	assert(get_rows() == m.get_rows());

	size_t tcols = get_cols();
	size_t mcols = m.get_cols();

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < get_rows(); i++) {
		total.clear();

		for (size_t j = 0; j < mcols; j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < get_rows(); i++) {
		for (size_t j = 0; j < tcols; j++)
			row[i].push_back(this->_array[i][j]);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_right(const Matrix &m)
{
	assert(get_rows() == m.get_rows());

	size_t tcols = get_cols();
	size_t mcols = m.get_cols();

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < get_rows(); i++) {
		total.clear();

		for (size_t j = 0; j < tcols; j++)
			total.push_back(this->_array[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < get_rows(); i++) {
		for (size_t j = 0; j < mcols; j++)
			row[i].push_back(m[i][j]);
	}

	return Matrix(row);
}

template <class T>
void Matrix <T> ::swap_rows(size_t a, size_t b)
{
	// Assumes that a and b are in bounds
	T *arr = &(this->_array[a * get_cols()]);
	T *brr = &(this->_array[b * get_cols()]);

	for (size_t i = 0; i < get_cols(); i++)
		std::swap(arr[i], brr[i]);
}

template <class T>
void Matrix <T> ::pow(const T &x)
{
	size_t s = get_cols() * get_rows();
	for (size_t i = 0; i < s; i++)
		this->_array[i] = std::pow(this->_array[i], x);
}

template <class T>
std::string dims(const Matrix <T> &a)
{
	return std::to_string(a.get_rows()) + " x " + std::to_string(a.get_cols());
}

// TODO: remove this method
template <class T>
std::string Matrix <T> ::display() const
{
        // TODO: STOP USING OSTREAM!
	std::ostringstream oss;

	oss << "[";

	for (size_t i = 0; i < get_rows(); i++) {
		if (get_cols() > 1) {
			oss << '[';

			for (size_t j = 0; j < get_cols(); j++) {
				oss << this->_array[i * get_cols() + j];
				if (j != get_cols() - 1)
					oss << ", ";
			}

			oss << ']';
		} else {
			oss << this->_array[i * get_cols()];
		}

		if (i < get_rows() - 1)
			oss << ", ";
	}

	oss << "]";

	return oss.str();
}

template <class T>
std::ostream &operator<<(std::ostream &os, const Matrix <T> &mat)
{

#ifdef ZHETAPI_DEBUG

	os << "(" << mat.get_rows() << " x " << mat.get_cols() << ") [";

#else

	os << "[";

#endif

	for (size_t i = 0; i < mat.get_rows(); i++) {
		if (mat.get_cols() > 1) {
			os << '[';

			for (size_t j = 0; j < mat.get_cols(); j++) {
				os << mat[i][j];
				if (j != mat.get_cols() - 1)
					os << ", ";
			}

			os << ']';
		} else {
			os << mat._array[i * mat.get_cols()];
		}

		if (i < mat.get_rows() - 1)
			os << ", ";
	}

	os << "]";

	return os;
}

}

#endif
