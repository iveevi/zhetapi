#ifndef INTERVAL_H_
#define INTERVAL_H_

// C/C++ headers
#include <iostream>
#include <iterator>
#include <set>
#include <vector>

namespace zhetapi {

namespace utility {

// N is the number of dimensions
template <size_t N = 1>
class Interval {
	template <size_t M>
	friend std::ostream &operator<<(std::ostream &, const Interval <M> &);
};

template <>
class Interval <1> {
	struct disjoint {
		using pflt = std::pair <double, double>;

		double	left	= 0;
		double	right	= 0;
		bool	closed	= true;

		double length() const {
			return right - left;
		}

		pflt approximate() const {
			static double epsilon = 1e-10;

			if (closed)
				return {left, right};

			return {left + epsilon, right - epsilon};
		}

		// The interval is completely to the left
		bool operator<(const disjoint &dj) const {
			pflt tapp = approximate();
			pflt oapp = dj.approximate();
			
			return (tapp.second < oapp.first)
				&& (tapp.first < oapp.first);
		}

		// The interval is completely to the right
		bool operator>(const disjoint &dj) const {
			pflt tapp = approximate();
			pflt oapp = dj.approximate();

			return (tapp.second > oapp.second)
				&& (tapp.first > oapp.second);
		}

		bool operator==(const disjoint &dj) const {
			return (left == dj.left)
				&& (right == dj.right)
				&& (closed == dj.closed);
		}
	};

	// Should always contain disjoint intervals
	std::set <disjoint>	__union;

	// Assumes that the intervals in un are disjoint
	explicit Interval(const std::set <disjoint> &un) : __union(un) {}
public:
	Interval(double left, double right, bool closed = true) {
		disjoint dj {left, right, closed};

		__union.insert(__union.begin(), dj);
	}

	double size() const {
		double len = 0;

		for (disjoint dj : __union)
			len += dj.length();
		
		return len;
	}

	friend Interval operator|(const Interval &, const Interval &);
	friend Interval operator&(const Interval &, const Interval &);

	friend std::ostream &operator<<(std::ostream &, const Interval &);
};

Interval <1> operator|(const Interval <1> &a, const Interval <1> &b)
{
	auto aset = a.__union;
	auto bset = b.__union;

	aset.insert(bset.begin(), bset.end());

	return Interval <1> (aset);
}

std::ostream &operator<<(std::ostream &os, const Interval <1> &itv)
{
	size_t sz = itv.__union.size();

	for (size_t i = 0; i < sz; i++) {
		auto itr = itv.__union.begin();

		std::advance(itr, i);

		if (itr->closed)
			os << "[";
		else
			os << "(";

		os << itr->left << ", " << itr->right;
		
		if (itr->closed)
			os << "]";
		else
			os << ")";

		if (i < sz - 1)
			os << " U ";
	}

	return os;
}

}

}

#endif
