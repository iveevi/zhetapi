#ifndef RANGE_H_
#define RANGE_H_

// Standard headers
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <type_traits>

namespace zhetapi {

// Range class
template <class T>
struct Range {
	T start;
	T term;
	T step;

	// Iterator type
	struct iterator {
		T value;
		T step;

		// Constructor
		iterator(T value, T step) : value(value), step(step) {}

		// Pre-increment
		iterator& operator++() {
			value += step;
			return *this;
		}

		// Post-increment
		iterator operator++(int) {
			iterator tmp(*this);
			value += step;
			return tmp;
		}

		// Pre-decrement
		iterator& operator--() {
			value -= step;
			return *this;
		}

		// Post-decrement
		iterator operator--(int) {
			iterator tmp(*this);
			value -= step;
			return tmp;
		}

		// Dereference
		T &operator*() {
			return value;
		}

		// Comparison operators
		bool operator==(const iterator &other) const {
			static T eps = T(100) * std::numeric_limits<T>::epsilon();
			if (std::is_floating_point <T> ::value)
				return std::fabs(value - other.value) < eps;
		
			return value == other.value;
		}

		bool operator!=(const iterator& other) const {
			return !(*this == other);
		}

		bool operator<(const iterator& other) const {
			return value < other.value;
		}

		bool operator<=(const iterator& other) const {
			return value <= other.value;
		}

		bool operator>(const iterator& other) const {
			return value > other.value;
		}

		bool operator>=(const iterator& other) const {
			return value >= other.value;
		}
	};

	// Constructors
	template <bool = std::is_arithmetic <T> ::value>
	Range(T pstart, T pterm, T pstep = T(1))
		: start(pstart), term(pterm), step(pstep) {}

	// Default is an empty range
	template <bool = std::is_arithmetic <T> ::value>
	Range(T pterm = T(0))
		: start(T(0)), term(pterm), step(T(1)) {}

	// Properties
	T length() const {
		return (term - start);
	}

	size_t size() const {
		return (size_t) (length() / step);
	}

	// Iterators
	iterator begin() const {
		return iterator(start, step);
	}

	iterator end() const {
		return iterator(term, step);
	}

	// Boolean operators
	// TODO: account for eps in comparison
	bool operator==(const Range& other) const {
		return (start == other.start)
			&& (term == other.term)
			&& (step == other.step);
	}

	bool operator!=(const Range& other) const {
		return !(*this == other);
	}

	// All the elements
	static const Range all;
};

// All for tensor slicing
extern Range <int> all;

// Printing
template <class T>
std::ostream &operator<<(std::ostream &os, const Range <T> &range)
{
	return os << "(" << range.start << ", " << range.step
		<< ", " << range.term << ")";
}

}

#endif
