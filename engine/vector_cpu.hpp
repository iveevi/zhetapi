template <class T>
Vector <T> ::Vector(const std::vector <T> &ref)
		: Matrix <T> (ref) {}

template <class T>
Vector <T> ::Vector(const std::initializer_list <T> &ref)
		: Vector(std::vector <T> (ref)) {}

template <class T>
template <class A>
Vector <T> ::Vector(const Vector <A> &other)
{
	if (is_vector_type <A> ()) {
		// Add a new function for this
		this->_array = new T[other.size()];
		this->_rows = other.get_rows();
		this->_cols = other.get_cols();

		this->_size = other.size();
		for (size_t i = 0; i < this->_size; i++)
			this->_array[i] = other[i];
		
		this->_dims = 1;
		this->_dim = new size_t[1];

		this->_dim[0] = this->_size;
	}
}

template <class T>
Vector <T> Vector <T> ::normalized() const
{
	std::vector <T> out;

	T dt = this->norm();

	for (size_t i = 0; i < size(); i++)
		out.push_back((*this)[i]/dt);

	return Vector(out);
}

template <class T>
Vector <T> &Vector <T> ::operator=(const Vector <T> &other)
{
	if (this != &other) {
		this->clear();

		this->_array = new T[other._size];
		this->_rows = other._rows;
		this->_cols = other._cols;

		this->_size = other._size;
		for (size_t i = 0; i < this->_size; i++)
			this->_array[i] = other._array[i];
		
		this->_dims = 1;
		this->_dim = new size_t[1];

		this->_dim[0] = this->_size;
	}

	return *this;
}

template <class T>
Vector <T> &Vector <T> ::operator=(const Matrix <T> &other)
{
	if (this != &other) {
		*this = Vector(other.get_rows(), T());

		for (size_t i = 0; i < this->_size; i++)
			this->_array[i] = other[0][i];
	}

	return *this;
}

template <class T>
void Vector <T> ::operator+=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] += a._array[i];
}

template <class T>
void Vector <T> ::operator-=(const Vector <T> &a)
{
	for (size_t i = 0; i < this->_size; i++)
		this->_array[i] -= a._array[i];
}

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
