template <class T>
Matrix <T> ::Matrix(const std::vector <Vector <T>> &columns)
{
	if (columns.size() > 0) {
		__rows = columns[0].size();
		__cols = columns.size();

		this->__size = __rows * __cols;

		this->__dim = new size_t[2];
		this->__dim[0] = __rows;
		this->__dim[1] = __cols;

		this->__array = new T[this->__size];

		for (size_t i = 0; i < __rows; i++) {
			for (size_t j = 0; j < __cols; j++)
				this->__array[__cols * i + j] = columns[j][i];
		}
	}
}

template <class T>
Matrix <T> ::Matrix(const std::initializer_list <Vector <T>> &columns)
		: Matrix(std::vector <Vector <T>> (columns)) {}

template <class T>
Matrix <T> ::Matrix(const std::vector <T> &ref) : Tensor <T> ({ref.size(), 1}, T())
{
	__rows = ref.size();

	assert(__rows > 0);

	__cols = 1;
	
	for (size_t i = 0; i < __rows; i++)
		this->__array[i] = ref[i];
}

template <class T>
Matrix <T> ::Matrix(const std::vector <std::vector <T>> &ref)
		: Tensor <T> (ref.size(), ref[0].size())
{
	__rows = ref.size();

	assert(__rows > 0);

	__cols = ref[0].size();

	assert(__cols > 0);
	
	for (int i = 0; i < __rows; i++) {
		for (int j = 0; j < __cols; j++) {
			assert(i < __rows && j < ref[i].size());
			
			this->__array[__cols * i + j] = ref[i][j];
		}
	}
}

template <class T>
Matrix <T> ::Matrix(const std::initializer_list <std::initializer_list <T>> &sq)
                : Tensor <T> (sq.size(), sq.begin()->size())
{
	__rows = sq.size();

	assert(__rows > 0);

	__cols = sq.begin()->size();

	assert(__cols > 0);
	
	size_t i = 0;
	for (auto lt : sq) {

		size_t j = 0;
		for (auto t : lt) {
			assert(i < __rows && j < lt.size());

			this->__array[__cols * i + (j++)] = t;
		}

		i++;
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T (size_t)> gen)
                : Tensor <T> (rs, cs)
{
	__rows = rs;
	__cols = cs;
	
	for (int i = 0; i < __rows; i++) {
		for (int j = 0; j < __cols; j++)
			this->__array[__cols * i + j] = gen(i);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T *(size_t)> gen)
                : Tensor <T> (rs, cs)
{
	__rows = rs;
	__cols = cs;
	
	for (int i = 0; i < __rows; i++) {
		for (int j = 0; j < __cols; j++)
			this->__array[__cols * i + j] = *gen(i);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T (size_t, size_t)> gen)
		: Tensor <T> (rs, cs)
{
	__rows = rs;
	__cols = cs;

	for (int i = 0; i < __rows; i++) {
		for (int j = 0; j < __cols; j++)
			this->__array[__cols * i + j] = gen(i, j);
	}
}

template <class T>
Matrix <T> ::Matrix(size_t rs, size_t cs, std::function <T *(size_t, size_t)> gen)
		: Tensor <T> (rs, cs)
{
	__rows = rs;
	__cols = cs;

	this->__array = new T[__rows * __cols];
	for (int i = 0; i < __rows; i++) {
		for (int j = 0; j < __cols; j++)
			this->__array[__cols * i + j] = *gen(i, j);
	}
}

template <class T>
void Matrix <T> ::write(std::ofstream &fout) const
{
	for (size_t i = 0; i < this->__size; i++)
		fout.write((char *) &(this->__array[i]), sizeof(T));
}

template <class T>
void Matrix <T> ::read(std::ifstream &fin)
{
	for (size_t i = 0; i < this->__size; i++)
		fin.read((char *) &(this->__array[i]), sizeof(T));
}

template <class T>
void Matrix <T> ::randomize(std::function <T ()> ftr)
{
	for (size_t i = 0; i < __rows; i++) {
		for (size_t j = 0; j < __cols; j++)
			this->__array[i * __cols + j] = ftr();
	}
}

template <class T>
Matrix <T> Matrix <T> ::append_above(const Matrix &m)
{
	assert(__cols == m.__cols);

	size_t t_rows = __rows;
	size_t m_rows = m.__rows;

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < m_rows; i++) {
		total.clear();

		for (size_t j = 0; j < __cols; j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < t_rows; i++) {
		total.clear();

		for (size_t j = 0; j < __cols; j++)
			total.push_back(this->__array[i][j]);

		row.push_back(total);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_below(const Matrix &m)
{
	assert(__cols == m.__cols);

	size_t t_rows = __rows;
	size_t m_rows = m.__rows;

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < t_rows; i++) {
		total.clear();

		for (size_t j = 0; j < __cols; j++)
			total.push_back(this->__array[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < m_rows; i++) {
		total.clear();

		for (size_t j = 0; j < __cols; j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_left(const Matrix &m)
{
	assert(__rows == m.__rows);

	size_t t_cols = __cols;
	size_t m_cols = m.__cols;

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < __rows; i++) {
		total.clear();

		for (size_t j = 0; j < m_cols; j++)
			total.push_back(m[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < __rows; i++) {
		for (size_t j = 0; j < t_cols; j++)
			row[i].push_back(this->__array[i][j]);
	}

	return Matrix(row);
}

template <class T>
Matrix <T> Matrix <T> ::append_right(const Matrix &m)
{
	assert(__rows == m.__rows);

	size_t t_cols = __cols;
	size_t m_cols = m.__cols;

	std::vector <std::vector <T>> row;

	std::vector <T> total;

	for (size_t i = 0; i < __rows; i++) {
		total.clear();

		for (size_t j = 0; j < t_cols; j++)
			total.push_back(this->__array[i][j]);

		row.push_back(total);
	}

	for (size_t i = 0; i < __rows; i++) {
		for (size_t j = 0; j < m_cols; j++)
			row[i].push_back(m[i][j]);
	}

	return Matrix(row);
}

template <class T>
void Matrix <T> ::swap_rows(size_t a, size_t b)
{
	// Assumes that a and b are in bounds
	T *arr = &(this->__array[a * __cols]);
	T *brr = &(this->__array[b * __cols]);

	for (size_t i = 0; i < __cols; i++)
		std::swap(arr[i], brr[i]);
}

template <class T>
void Matrix <T> ::pow(const T &x)
{
	size_t s = __cols * __rows;
	for (size_t i = 0; i < s; i++)
		this->__array[i] = std::pow(this->__array[i], x);
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

	for (int i = 0; i < __rows; i++) {
		if (__cols > 1) {
			oss << '[';

			for (int j = 0; j < __cols; j++) {
				oss << this->__array[i * __cols * j];
				if (j != __cols - 1)
					oss << ", ";
			}

			oss << ']';
		} else {
			oss << this->__array[i * __cols];
		}

		if (i < __rows - 1)
			oss << ", ";
	}

	oss << "]";

	return oss.str();
}

template <class T>
std::ostream &operator<<(std::ostream &os, const Matrix <T> &mat)
{
	os << "[";

	for (size_t i = 0; i < mat.__rows; i++) {
		if (mat.__cols > 1) {
			os << '[';

			for (size_t j = 0; j < mat.__cols; j++) {
				os << mat[i][j];
				if (j != mat.__cols - 1)
					os << ", ";
			}

			os << ']';
		} else {
			os << mat.__array[i * mat.__cols];
		}

		if (i < mat.__rows - 1)
			os << ", ";
	}

	os << "]";

	return os;
}