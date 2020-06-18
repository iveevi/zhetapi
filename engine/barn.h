#ifndef BARN_H_
#define BARN_H_

#include <iostream>
#include <string>

#include "table.h"
#include "rational.h"
#include "element.h"
#include "matrix.h"
#include "zcomplex.h"

/**
 * @brief Represents the working
 * space of a zhetapi function
 * or application; the sets of
 * integer, real, complex, rational,
 * vector and matrix fields.
 *
 * @tparam R The type of a scalar
 * real value. Defaults to
 * [long double].
 *
 * @tparam Z The type of a scalar
 * integer value (used in rationals).
 * Defaults to [long long int].
 */
template <class T = long double, class U = long long int>
class barn {
public:
	using Z = scalar <U>;
	using R = scalar <T>;

	using Q = rational <U>;

	using RC = zcomplex <T>;
	using QC = zcomplex <Q>;

	using RM = matrix <T>;		// These matrix fields also include
	using QM = matrix <Q>;		// vector spaces of the corresponding sets
private:
	table <Z> z;
	table <R> r;
	table <Q> q;

	table <RC> rc;
	table <QC> qc;

	table <RM> rm;
	table <QM> qm;
public:
	barn();

	void put_z(const Z &);
	void put_r(const R &);
	void put_q(const Q &);
	void put_rc(const RC &);
	void put_qc(const QC &);
	void put_rm(const RM &);
	void put_qm(const QM &);

	void print();
};

template <class R, class Z>
barn <R, Z> ::barn() : z(), r(), q(), rc(), qc(), rm(), qm() {}

template <class R, class Z>
void barn <R, Z> ::print()
{
	std::cout << std::string(50, '=') << endl;
	std::cout << "INTEGERS:" << endl;
	std::cout << std::string(50, '=') << endl;

	z.print();

	std::cout << std::string(50, '=') << endl;
	std::cout << "REALS:" << endl;
	std::cout << std::string(50, '=') << endl;

	r.print();
	
	std::cout << std::string(50, '=') << endl;
	std::cout << "RATIONALS:" << endl;
	std::cout << std::string(50, '=') << endl;

	q.print();
	
	std::cout << std::string(50, '=') << endl;
	std::cout << "REAL COMPLEX:" << endl;
	std::cout << std::string(50, '=') << endl;

	rc.print();

	std::cout << std::string(50, '=') << endl;
	std::cout << "RATIONAL COMPLEX:" << endl;
	std::cout << std::string(50, '=') << endl;

	qc.print();
	
	std::cout << std::string(50, '=') << endl;
	std::cout << "REAL MATRICES:" << endl;
	std::cout << std::string(50, '=') << endl;

	rm.print();
	
	std::cout << std::string(50, '=') << endl;
	std::cout << "RATIONAL MATRICES:" << endl;
	std::cout << std::string(50, '=') << endl;

	qm.print();
}

#endif
