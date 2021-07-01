#ifndef MATRIX_CPU_H_
#define MATRIX_CPU_H_

namespace zhetapi {

template <class T>
Matrix <T> ::Matrix(const std::vector <Vector <T>> &columns)
{
	if (columns.size() > 0) {
		_rows = columns[0].size();
		_cols = columns.size();

		this->_size = _rows * _cols;

		this->_dim = new size_t[2];
		this->_dim[0] = _rows;
		this->_dim[1] = _cols;

		this->_array = new T[this->_size];

		for (size_t i = 0; i < _rows; i++) {
			for (size_t j = 0; j < _cols; j++)
				this->_array[_cols * i + j] = columns[j][i];
		}
	}
}

template <class T>
Matrix <T> ::Matrix(const std::initializer_list <Vector <T>> &columns)
		: Matrix(std::vector <Vector <T>> (columns)) {}

template <class T>
Matrix <T> ::Matrix(const std::vector <T> &ref) : Tensor <T> ({ref.size(), 1}, T())
{
	_rows = ref.size();

	assert(_rows > 0);

	_cols = 1;
	
	for (size_t i = 0; i < _rows; i++)
		this->_array[i] = ref[i];
}

template <class T>
Matrix <T> ::Matrix(const std::vector <std::vector <T>> &ref)
		: Tensor <T> (ref.size(), ref[0].size())
{
	_rows = ref.size();

	assert(_rows > 0);

	_cols = ref[0].size();

	assert(_cols > 0);
	
	for (int i = 0; i < _rows; i++) {
		for (int j = 0; j < _cols; j++) {
			assert(i < _rows && j < ref[i].size());
			
			this->_array[_cols * i + j] = ref[i][j];
		}
	}
}

template <class T>
Matrix <T> ::Matrix(const std::initializer_list <std::initializer_list <T>> &sq)
                : Tensor <T> (sq.size(), sq.begin()->size())
{
	_rows = sq.size();

	assert(_rows > 0);

	_cols = sq.begin()->size();

	assert(_cols > 0);
	
	size_t i = 0;
	for (auto lt : sq) {

		size_t j = 0;
		for (auto t : lt) {
			assert(i < _rows && j < lt.size());

			this->_array[_cols * i + (j++)] = t;
		}

		i++;
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T (size_t)> gen)
                : _rows(rs), _cols(cs), Tensor <T> (rs, cs)
{	
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[_cols * i + j] = gen(i);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T *(size_t)> gen)
                : _rows(rs), _cols(cs), Tensor <T> (rs, cs)
{
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[_cols * i + j] = *gen(i);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T (size_t, size_t)> gen)
		: Tensor <T> (rs, cs), _rows(rs), _cols(cs)
{
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[_cols * i + j] = gen(i, j);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T *(size_t, size_t)> gen)
		: Tensor <T> (rs, cs)
{
	_rows = rs;
	_cols = cs;

	this->_array = new T[_rows * _cols];
	for (int i = 0; i < _rows; i++) {
		for (int j = 0; j < _cols; j++)
			this->_array[_cols * i + j] = *gen(i, j);
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
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			this->_array[i * _cols + j] = ftr();
	}
}

template <class T>
Matrix <T> Matrix <T> ::append_above(const Matrix &m)
{
	assert(_cols == m._cols);

	size_t t_rows = _rows;
	size_t m_rows = m._rows;

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < m_rows; i++) {
		total.clear();

		for (size_t j = 0; j < _cols; j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < t_rows; i++) {
		total.clear();

		for (size_t j = 0; j < _cols; j++)
			total.push_back(this->_array[i][j]);

		row.push_back(total);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_below(const Matrix &m)
{
	assert(_cols == m._cols);

	size_t t_rows = _rows;
	size_t m_rows = m._rows;

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < t_rows; i++) {
		total.clear();

		for (size_t j = 0; j < _cols; j++)
			total.push_back(this->_array[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < m_rows; i++) {
		total.clear();

		for (size_t j = 0; j < _cols; j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_left(const Matrix &m)
{
	assert(_rows == m._rows);

	size_t t_cols = _cols;
	size_t m_cols = m._cols;

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < _rows; i++) {
		total.clear();

		for (size_t j = 0; j < m_cols; j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < t_cols; j++)
			row[i].push_back(this->_array[i][j]);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_right(const Matrix &m)
{
	assert(_rows == m._rows);

	size_t t_cols = _cols;
	size_t m_cols = m._cols;

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < _rows; i++) {
		total.clear();

		for (size_t j = 0; j < t_cols; j++)
			total.push_back(this->_array[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < m_cols; j++)
			row[i].push_back(m[i][j]);
	}

	return Matrix(row);
}

template <class T>
void Matrix <T> ::swap_rows(size_t a, size_t b)
{
	// Assumes that a and b are in bounds
	T *arr = &(this->_array[a * _cols]);
	T *brr = &(this->_array[b * _cols]);

	for (size_t i = 0; i < _cols; i++)
		std::swap(arr[i], brr[i]);
}

template <class T>
void Matrix <T> ::pow(const T &x)
{
	size_t s = _cols * _rows;
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

	for (size_t i = 0; i < _rows; i++) {
		if (_cols > 1) {
			oss << '[';

			for (size_t j = 0; j < _cols; j++) {
				oss << this->_array[i * _cols + j];
				if (j != _cols - 1)
					oss << ", ";
			}

			oss << ']';
		} else {
			oss << this->_array[i * _cols];
		}

		if (i < _rows - 1)
			oss << ", ";
	}

	oss << "]";

	return oss.str();
}

template <class T>
std::ostream &operator<<(std::ostream &os, const Matrix <T> &mat)
{
	os << "[";

	for (size_t i = 0; i < mat._rows; i++) {
		if (mat._cols > 1) {
			os << '[';

			for (size_t j = 0; j < mat._cols; j++) {
				os << mat[i][j];
				if (j != mat._cols - 1)
					os << ", ";
			}

			os << ']';
		} else {
			os << mat._array[i * mat._cols];
		}

		if (i < mat._rows - 1)
			os << ", ";
	}

	os << "]";

	return os;
}

}

#endif