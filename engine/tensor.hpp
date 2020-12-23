#ifndef TENSOR_H_
#define TENSOR_H_

// C/C++ headers
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace zhetapi {

	template <class T>
	class Tensor {
        protected:
		T		*__array;
		size_t		__size;

		// Variables for printing
		size_t		*__dim;
		size_t		__dims;

#ifdef ZHP_CUDA

		bool		__on_device;

#endif

	public:
		Tensor(const ::std::vector <::std::size_t> &, const ::std::vector <T> &);
		
		// Printing functions
		::std::string print() const;

		template <class U>
		friend ::std::ostream &operator<<(::std::ostream &, const Tensor <U> &);

                // Dimension mismatch exception
                class dimension_mismatch {};
		class bad_dimensions {};

#ifndef ZHP_CUDA

                // Construction and memory
		Tensor();
                Tensor(const Tensor &);
		Tensor(const ::std::vector <T> &);
		Tensor(const ::std::vector <::std::size_t> &, const T & = T());

		~Tensor();

                Tensor &operator=(const Tensor &);

		// Indexing
		T &operator[](const ::std::vector <size_t> &);
		const T &operator[](const ::std::vector <size_t> &) const;

		// Comparison
		template <class U>
		friend bool operator==(const Tensor <U> &, const Tensor <U> &);

#else

		__host__ __device__
		Tensor();

		__host__ __device__
		Tensor(bool);

		__host__ __device__
                Tensor(const Tensor &);
		
		__host__ __device__
		Tensor(size_t, size_t, const T &);

		__host__ __device__
		Tensor(const ::std::vector <T> &);

		__host__ __device__
		Tensor(const ::std::vector <::std::size_t> &, const T & = T());

		__host__ __device__
		~Tensor();

		__host__ __device__
                Tensor &operator=(const Tensor &);

		__host__ __device__
		void set_device_status(bool);

		template <class U>
		__host__ __device__
		friend bool operator==(const Tensor <U> &, const Tensor <U> &);

#endif

	};

	template <class T>
	Tensor <T> ::Tensor(const ::std::vector <size_t> &dim, const ::std::vector <T> &arr)
			: __dims(dim.size())
	{

#ifdef ZHP_CUDA

		__on_device = false;

#endif

		__dim = new size_t[__dims];

		size_t prod = 1;
		for (size_t i = 0; i < __dims; i++) {
			prod *= dim[i];

			__dim[i] = dim[i];
		}

		__size = prod;

		if (!__size)
			throw bad_dimensions();

		if (arr.size() != __size)
                        throw dimension_mismatch();

		__array = new T[prod];

		for (size_t i = 0; i < prod; i++)
			__array[i] = arr[i];
	}

	// Printing functions
	template <class T>
	::std::string Tensor <T> ::print() const
	{
		if (!__dim)
			return "[]";
		
		if (__dims == 0) {
			::std::ostringstream oss;

			oss << __array[0];

			return oss.str();
		}
		
		::std::string out = "[";

		::std::vector <size_t> cropped;
		for (int i = 0; i < ((int) __dims) - 1; i++)
			cropped.push_back(__dim[i + 1]);

		size_t left = __size/__dim[0];
		for (size_t i = 0; i < __dim[0]; i++) {
			::std::vector <T> elems;

			for (size_t k = 0; k < left; k++)
				elems.push_back(__array[left * i + k]);
			
			Tensor tmp(cropped, elems);

			out += tmp.print();

			if (i < __dim[0] - 1)
				out += ", ";
		}

		return out + "]";
	}

	template <class T>
	::std::ostream &operator<<(::std::ostream &os, const Tensor <T> &ts)
	{
		os << ts.print();

		return os;
	}

#ifndef ZHP_CUDA

	// Constructors and memory relevant functions
	template <class T>
	Tensor <T> ::Tensor() : __dim(nullptr), __dim(0), __array(nullptr),
			__size(0) {}

        template <class T>
        Tensor <T> ::Tensor(const Tensor <T> &other) : __dims(other.__dims), __size(other.__size)
        {
                __dim = new size_t[__dims];
                for (size_t i = 0; i < __dims; i++)
                        __dim[i] = other.__dim[i];

                __array = new T[__size];
                for (size_t i = 0; i < __size; i++)
                        __array[i] = other.__array[i];
        }

	template <class T>
	Tensor <T> ::Tensor(const ::std::vector <T> &arr) : __dims(1), __size(arr.size())
	{
		__dim = new size_t[1];

		__dim[0] = __size;

		if (!__size)
			throw bad_dimensions();

		__array = new T[__size];

		for (size_t i = 0; i < __size; i++)
			__array[i] = arr[i];
	}

	template <class T>
	Tensor <T> ::Tensor(const ::std::vector <size_t> &dim, const T &def)
			: __dims(dim.size())
	{
		__dim = new size_t[__dims];

		size_t prod = 1;
		for (size_t i = 0; i < __dims; i++) {
			prod *= dim[i];

			__dim[i] = dim[i];
		}

		__size = prod;

		if (!__size)
			throw bad_dimensions();

		__array = new T[prod];

		for (size_t i = 0; i < prod; i++)
			__array[i] = def;
	}

	template <class T>
	Tensor <T> ::~Tensor()
	{
		delete[] __dim;
		delete[] __array;
	}

        template <class T>
        Tensor <T> &Tensor <T> ::operator=(const Tensor <T> &other)
        {
                if (this != &other) {
                        __dims = other.__dims;
                        __size = other.__size;
                
                        __dim = new size_t[__dims];
                        for (size_t i = 0; i < __dims; i++)
                                __dim[i] = other.__dim[i];

                        __array = new T[__size];
                        for (size_t i = 0; i < __size; i++)
                                __array[i] = other.__array[i];
                }

                return *this;
        }

	// Index
	template <class T>
	T &Tensor <T> ::operator[](const ::std::vector <size_t> &indices)
	{
		size_t full = 0;

		assert(indices.size() == __dims);
		for (size_t i = 0; i < __dims; i++)
			full += indices[i] * __dim[__dims - (i + 1)];
		
		return __array[full];
	}

	template <class T>
	const T &Tensor <T> ::operator[](const ::std::vector <size_t> &indices) const
	{
		size_t full = 0;

		assert(indices.size() == __dims);
		for (size_t i = 0; i < __dims; i++)
			full += indices[i] * __dim[__dims - (i + 1)];
		
		return __array[full];
	}

	// Comparison
	template <class T>
	bool operator==(const Tensor <T> &a, const Tensor <T> &b)
	{
		if (a.__size != b.__size)
			return false;

		for (size_t i = 0; i < a.__size; i++) {
			if (a.__array[i] != b.__array[i])
				return false;
		}

		return true;
	}

#endif

}

#endif
