#ifndef INTERVAL_H_
#define INTERVAL_H_

// TODO: move out of std

// Essentials
#include "../avr/essentials.hpp"

#ifdef __AVR	// Does not support AVR

#include "../avr/random.hpp"

#else

// C/C++ headers
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <vector>

#endif		// Does not support AVR

namespace zhetapi {

namespace utility {

#ifdef __AVR

// Blank dummy struct
struct random_generator {};

// Cite Ran from Numerical Recipes
struct distro_engine {
	long double operator()(const random_generator &rgen) {
		static avr::RandomEngine reng(16183LL);

		return reng.ldouble();
	}
};

using dre = random_generator;
using udb = distro_engine;

#else

// Typedefs for sanity
using dre = std::default_random_engine;
using udb = std::uniform_real_distribution <double>;

#endif

// N is the number of dimensions
template <size_t N = 1>
class Interval {
	AVR_IGNORE(
		template <size_t M>
		friend std::ostream &operator<<(std::ostream &, const Interval <M> &)
	);
};

// Keep inlined here for header only purposes
struct disjoint {
	static dre gen;
	static udb distro;

#ifdef __AVR

	using pflt = _avr_pair <double, double>;

#else

	using pflt = std::pair <double, double>;

#endif

	double left = 0;
	double right = 0;
	bool closed = true;

	disjoint(double l, double r, bool c)
			: left(l), right(r), closed(c) {}

	double length() const
	{
		return right - left;
	}

	pflt approximate() const
	{
		static double epsilon = 1e-10;

		if (closed)
			return {left, right};

		return {left + epsilon, right - epsilon};
	}

	// Use a real uniform distro later
	double uniform() const
	{
		return left + distro(gen) * (right - left);
	}

	// Check disjointed-ness
	bool is_disjoint(const disjoint &dj) const
	{
		// If either interval is greater,
		// then it must be disjoint
		return (*this > dj) || (*this < dj);
	}

	// The interval is completely to the left
	bool operator<(const disjoint &dj) const
	{
		pflt tapp = approximate();
		pflt oapp = dj.approximate();

		return (tapp.second < oapp.first) && (tapp.first < oapp.first);
	}

	// The interval is completely to the right
	bool operator>(const disjoint &dj) const
	{
		pflt tapp = approximate();
		pflt oapp = dj.approximate();

		return (tapp.second > oapp.second) && (tapp.first > oapp.second);
	}

	bool operator==(const disjoint &dj) const
	{
		return (left == dj.left) && (right == dj.right) && (closed == dj.closed);
	}
};

AVR_MASK(dre disjoint::gen = dre());
AVR_MASK(udb disjoint::distro = udb());

#ifndef __AVR	// Does not support AVR

// TODO: Switch from double to long double
template <>
class Interval <1> {
	// For random generation
	static dre gen;
	static udb distro;

	// Should always contain disjoint intervals
	std::set <disjoint>	_union;

	// Assumes that the intervals in un are disjoint
	explicit Interval(const std::set <disjoint> &un) : _union(un) {}

	// Checks that the new 'disjoint' interval is indeed disjoint
	bool is_disjoint(const disjoint &djx) const {
		for (const disjoint &dj : _union) {
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

		_union.insert(_union.begin(), dj);
	}

	// Properties
	double size() const {
		double len = 0;

		for (disjoint dj : _union)
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

		double *db = new double[_union.size() + 1];

		size_t i = 0;

		db[i++] = 0;
		for (disjoint dj : _union) {
			db[i] = db[i - 1] + dj.length()/len;

			i++;
		}

		double rnd = distro(gen);

		for (i = 0; i < _union.size(); i++) {
			if ((rnd > db[i]) && (rnd < db[i + 1]))
				break;
		}

		delete[] db;

		auto itr = _union.begin();

		std::advance(itr, i);

		return itr->uniform();
	}

	// Operations
	Interval &operator|=(const Interval &itv) {
		auto iset = itv._union;

		using namespace std;

		// Check for disjointed-ness
		for (const disjoint &dj : iset) {
			if (is_disjoint(dj))
				_union.insert(_union.begin(), dj);
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

#else		// AVR support

#warning Zhetapi does not support the all of zhetapi::utility::Interval for AVR systems.

// TODO: Add a singular disjoint
template <>
class Interval <1> {
	disjoint dj;
public:
	// Defaults to [0, 1]
	Interval() : Interval(1.0L) {}

	explicit Interval(unsigned long long int x)
			: Interval((long double) x) {}

	explicit Interval(long double x)
			: Interval(0, x) {}

	Interval(double left, double right, bool closed = true)
			: dj(left, right, closed) {}

	double size() const {
		return dj.length();
	}

	operator bool() const {
		return size() > 0;
	}
};

#endif		// AVR support switch

}

}

#endif
