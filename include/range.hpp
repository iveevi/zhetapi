#ifndef RANGE_H_
#define RANGE_H_

// Standard headers
#include <cstdint>
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
		// TODO: account for eps in comparison
		bool operator==(const iterator& other) const {
			return value == other.value;
		}

		bool operator!=(const iterator& other) const {
			return value != other.value;
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

	template <bool = std::is_arithmetic <T> ::value>
	Range(T pterm)
		: start(T(0)), term(pterm), step(T(1)) {}

	// Iterators
	iterator begin() const {
		return iterator(start, step);
	}

	iterator end() const {
		return iterator(term, step);
	}
};

}

#endif
