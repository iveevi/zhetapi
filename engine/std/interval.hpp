#ifndef INTERVAL_H_
#define INTERVAL_H_

// C/C++ headers
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <vector>

namespace zhetapi {

namespace utility {

// Typedefs for sanity
using dre = std::default_random_engine;
using udb = std::uniform_real_distribution <double>;

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
	static dre gen;
	static udb distro;

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
			return left + distro(gen) * (right - left);
		}

		// Check disjointed-ness
		bool is_disjoint(const disjoint &dj) const {
			// If either interval is greater,
			// then it must be disjoint
			return (*this > dj) || (*this < dj);
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

	// Checks that the new 'disjoint' interval is indeed disjoint
	bool is_disjoint(const disjoint &djx) const {
		for (const disjoint &dj : __union) {
			if (!dj.is_disjoint(djx))
				return false;
		}

		return true;
	}
public:
	// Defaults to [0, 1]
	Interval() : Interval(1.0L) {}

	explicit Interval(unsigned long long int x)
			: Interval((long double) x) {}

	explicit Interval(long double x)
			: Interval(0, x) {}

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

	operator bool() const {
		return size() > 0;
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

		double rnd = distro(gen);

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

		using namespace std;

		// Check for disjointed-ness
		for (const disjoint &dj : iset) {
			if (is_disjoint(dj))
				__union.insert(__union.begin(), dj);
			else
				cout << "Adding a non-disjoint interval" << endl;
		}

		return *this;
	}

	// Binary operations
	friend Interval operator|(const Interval &, const Interval &);
	friend Interval operator&(const Interval &, const Interval &);

	friend std::ostream &operator<<(std::ostream &, const Interval &);
};

Interval <1> operator|(const Interval <1> &, const Interval <1> &);

std::ostream &operator<<(std::ostream &, const Interval <1> &);

// Literal constructor
Interval <1> operator""_I(unsigned long long int);
Interval <1> operator""_I(long double);

}

}

#endif
