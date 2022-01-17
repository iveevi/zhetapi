#ifndef INTERVAL_H_
#define INTERVAL_H_

// C/C++ headers
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <vector>

// Engine headers
#include "../fixed_vector.hpp"

namespace zhetapi {

// TODO: inspect the random-ness of the interval class
namespace utility {

extern std::random_device rd;

// Typedefs for sanity
using dre = std::mt19937;
using udb = std::uniform_real_distribution <double>;

// TODO: extend to long double

// Keep inlined here for header only purposes
struct disjoint {
	static dre gen;
	static udb distro;

	using pflt = std::pair <double, double>;

	double left = 0;
	double right = 0;
	bool closed = true;

	// c should represent compact instead of closed?
	disjoint(double l = 0.0, double r = 0.0, bool c = true)
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

// N is the number of dimensions
// NOTE: for now multidim intervals
// can only be one "box", see the TODO
// below

/**
 * @brief Random generator class, that can uniformly generated Vectors (or
 * scalars) with elements that are randomly sampled from a distribution
 * (currently each element can be sampled only from a uniform distribution
 * that is the union of disjoint intervals).
 *
 * @tparam N the dimension that the corresponding random Vectors should have.
 */
template <size_t N = 1>
class Interval {
	// TODO: this will not work,
	// we need a disjoint equivalent for N dimensions
	// (think about boxes as N-dim intervals)
	disjoint *axes = nullptr;
public:
	Interval() : Interval(1.0L) {}

	Interval(long double x) {
		axes = new disjoint[N];

		for (size_t i = 0; i < N; i++)
			axes[i] = disjoint(0, x, true);
	}

	FixedVector <double, N> operator()() const {
		return uniform();
	}

	FixedVector <double, N> uniform() const {
		// First check that the axes are not null
		if (axes == nullptr)
			throw null_axes();

		return FixedVector <double, N> (
			[&](size_t i) -> double {
				return axes[i].uniform();
			}, N
		);
	}

	template <size_t M>
	friend std::ostream &operator<<(std::ostream &,
		const Interval <M> &);

	// Exceptions
	class null_axes : public std::runtime_error {
	public:
		null_axes() : std::runtime_error("Axes of Interval <N>"
			" are null") {}
	};
};

// TODO: Switch from double to long double
/**
 * @brief Single dimensional (scalar) random generator. Can sample uniformly
 * from a union of intervals.
 */
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

	double operator()() const {
		return uniform();
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

}

}

#endif
