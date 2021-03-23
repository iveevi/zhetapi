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
		this->__array = new T[other.size()];
		this->__rows = other.get_rows();
		this->__cols = other.get_cols();

		this->__size = other.size();
		for (size_t i = 0; i < this->__size; i++)
			this->__array[i] = other[i];
		
		this->__dims = 1;
		this->__dim = new size_t[1];

		this->__dim[0] = this->__size;
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
