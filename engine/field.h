#ifndef FIELD_H_
#define FIELD_H_

#include <iostream>
#include <string>

#include "table.h"
#include "rational.h"
#include "element.h"
#include "matrix.h"
#include "zcomplex.h"

enum set_id {
	s_real,
	s_rational
};

template <class U>
struct number {
	U _dat;

	operator U() const;
};

template <class U>
number ::operator U() const
{
	return _dat;
}

/**
 * @brief Represents the working
 * space of a zhetapi function
 * or application; the sets of
 * real, complex, rational, vector
 * and matrix fields.
 *
 * @tparam R The type of a scalar
 * real value. Defaults to
 * [long double].
 *
 * @tparam Z The type of a scalar
 * integer value (used in rationals).
 * Defaults to [long long int].
 */
template <class R = long double, class Z = long long int>
class field {
public:
	using Q = rational <Z>;

	using RC = zcomplex <R>;
	using QC = zcomplex <Q>;

	using RM = matrix <R>;
	using QM = matrix <Q>;
private:
	table <R> r;
	table <Q> q;

	table <RC> rc;
	table <QC> qc;

	table <RM> rm;
	table <QM> qm;
public:
	field();

	void print();
};

template <class R, class Z>
field <R, Z> ::field() : r(), q(), rc(), qc(), rm(), qm() {}

template <class R, class Z>
void field <R, Z> ::print()
{
	std::cout << std::string(50, '=') << endl;
	std::cout << "REALS:" << endl;
	std::cout << std::string(50, '=') << endl;

	r.print();
	
	std::cout << std::string(50, '=') << endl;
	std::cout << "RATIONALS:" << endl;
	std::cout << std::string(50, '=') << endl;

	q.print();
}

#endif
