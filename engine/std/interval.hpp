#ifndef INTERVAL_H_
#define INTERVAL_H_

// C/C++ headers
#include <iostream>
#include <iterator>
#include <set>
#include <vector>

namespace zhetapi {

namespace utility {

// Some arbitrary random utils

// Use an actual uniform distro
auto runit = []() {
	srand(clock());

	return (rand() / ((double) RAND_MAX));
};

// N is the number of dimensions
template <size_t N = 1>
class Interval {
	template <size_t M>
	friend std::ostream &operator<<(std::ostream &, const Interval <M> &);
};

// TODO: Switch from double to long double
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

		// Use a real uniform distro later
		double uniform() const {
			return left + runit() * (right - left);
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
	Interval(unsigned long long int x) : Interval((long double) x) {}
	Interval(long double x) : Interval(0, x) {}

	Interval(double left, double right, bool closed = true) {
		disjoint dj {left, right, closed};

		__union.insert(__union.begin(), dj);
	}

	// Properties
	double size() const {
		double len = 0;

		for (disjoint dj : __union)
			len += dj.length();
		
		return len;
	}

	// Sampling
	double uniform() const {
		// TODO: Cover case where the interval is not closed
		double len = size();

		double *db = new double[__union.size() + 1];

		size_t i = 0;

		db[i++] = 0;
		for (disjoint dj : __union) {
			db[i] = db[i - 1] + dj.length()/len;

			i++;
		}

		double rnd = runit();

		for (i = 0; i < __union.size(); i++) {
			if ((rnd > db[i]) && (rnd < db[i + 1]))
				break;
		}

		delete[] db;

		auto itr = __union.begin();

		std::advance(itr, i);

		return itr->uniform();
	}

	// Operations
	Interval &operator|=(const Interval &itv) {
		auto iset = itv.__union;

		// Check for disjointed-ness
		__union.insert(iset.begin(), iset.end());

		return *this;
	}

	// Binary operations
	friend Interval operator|(const Interval &, const Interval &);
	friend Interval operator&(const Interval &, const Interval &);

	friend std::ostream &operator<<(std::ostream &, const Interval &);
};

Interval <1> operator|(const Interval <1> &a, const Interval <1> &b)
{
	Interval <1> out = a;

	return out |= b;
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

// Literal constructor
Interval <1> operator""_I(unsigned long long int x)
{
	return Interval <1> (x);
}

Interval <1> operator""_I(long double x)
{
	return Interval <1> (x);
}

}

}

#endif