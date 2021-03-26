#ifndef SLIDING_BUFFER_H_
#define SLIDING_BUFFER_H_

namespace zhetapi {

namespace ads {

template <class T, class F>
class SlidingBuffer {
	T *	_array	= nullptr;

	T	_avg	= 0;

	size_t	_size	= 0;
	size_t	_index	= 0;
public:
	SlidingBuffer();
	SlidingBuffer(const SlidingBuffer &);
	
	explicit SlidingBuffer(size_t);

	SlidingBuffer &operator=(const SlidingBuffer &);

	~SlidingBuffer();

	void resize(size_t);

	void insert(const T &);

	T ipop(const T &);
	const T &iavg(const T &);

	const T &avg() const;
};

template <class T, class F>
SlidingBuffer <T, F> ::SlidingBuffer() {}

template <class T, class F>
SlidingBuffer <T, F> ::SlidingBuffer(const SlidingBuffer &other)
		: _size(other._size), _index(other._index),
		_avg(other._avg)
{
	_array = new T[_size];

	memcpy(_array, other._array, sizeof(T) * _size);
}

template <class T, class F>
SlidingBuffer <T, F> ::SlidingBuffer(size_t buffer_size)
	: _size(buffer_size)
{
	_array = new T[_size];
}

template <class T, class F>
SlidingBuffer <T, F> &SlidingBuffer <T, F>
		::operator=(const SlidingBuffer &other)
{
	if (this != &other) {
		if (_array)
			delete[] _array;

		_size = other._size;
		_index = other._index;
		_avg = other._index;

		_array = new T[_size];

		memcpy(_array, other._array, sizeof(T) * _size);
	}

	return *this;
}

template <class T, class F>
SlidingBuffer <T, F> ::~SlidingBuffer()
{
	delete[] _array;
}

template <class T, class F>
void SlidingBuffer <T, F> ::resize(size_t size)
{
	T tmp = new T[size];

	memcpy(tmp, _array, std::min(size, _size) * sizeof(T));

	for (size_t i = _size; i < _size - size; i++)
		_Avg -= F(_array[i]);

	index %= size;

	_size = size;
}

template <class T, class F>
void SlidingBuffer <T, F> ::insert(const T &x)
{
	_avg += F(x) - F(_array[i]);

	_array[_index] = x;

	_index = (_index + 1) % _size;
}

template <class T, class F>
const T &SlidingBuffer <T, F> ::iavg(const T &x) const
{
	insert(x);

	return _avg;
}

template <class T, class F>
T SlidingBuffer <T, F> ::ipop(const T &x)
{
	T tmp = _avg[i];

	_avg += F(x) - F(tmp);

	_array[_index] = x;

	_index = (_index + 1) % _size;

	return tmp;
}

template <class T, class F>
const T &SlidingBuffer <T, F> ::avg() const
{
	return _avg;
}

}

}

#endif
