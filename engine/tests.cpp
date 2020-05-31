#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <gmpxx.h>

#include "config.h"
#include "expression.h"
#include "matrix.h"
#include "stree.h"
#include "table.h"
#include "variable.h"
#include "element.h"
#include "utility.h"

using namespace std;

void test_variable()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING VARIABLE TEST" << endl;
	cout << string('#', 100) << endl;

	variable <double> x("x", false, 12);

	cout << "Initial Value of Varaible: " << *x << endl;

	x[323.0];

	cout << "Value is Now: " << *x << endl;
}

void test_expression()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING EXPRESSION TEST" << endl;
	cout << string('#', 100) << endl;

	table <double> tbl;

	vector <variable <double>> vals {
		variable <double> {"pi", acos(-1)},
		variable <double> {"e", exp(1)}
	};
	
	for (variable <double> v : vals)
		tbl.insert_var(v);

	cout << "Expression evaluation with format: "
		<< expression <double> ::in_place_evaluate_formatted("3e^3 * sin(cos(pi/2))")
		<< endl;

	string input;
	while (true) {
		cout << "Enter an expression to be evaluated: ";
		getline(cin, input);

		cout << "\t" << expression <double>
			::in_place_evaluate(input) << endl;
	}
}
void test_variable_parsing()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING VARIABLE PARSING TEST" << endl;
	cout << string('#', 100) << endl;

	table <double> tbl;

	vector <variable <double>> vals {variable <double> {"x", 12.0},
		variable <double> {"ran", true, 123},
		variable <double> {"y", 13423.423},
		variable <double> {"this", true, 12.0},
		variable <double> {"pi", acos(-1)},
		variable <double> {"e", exp(1)}
	};
	
	for (variable <double> v : vals) {
		// cout << "Inserting: " << v << endl;
		tbl.insert_var(v);
	}

	cout << endl;

	string input = "3e^3 * sin(cos(pi/2))";
	cout << "Expression [" << input << "] = " << expression <double>
		::in_place_evaluate(input, tbl) << endl;
	
	while (true) {
		cout << "Enter an expression to be evaluated: ";
		getline(cin, input);

		cout << "\t" << expression <double>
			::in_place_evaluate(input, tbl) << endl;
	}
}

void test_function()
{
	cout << endl << string(100, '#') << endl;
	cout << "BEGINNING FUNCTION TEST" << endl;
	cout << string(100, '#') << endl;

	/* clock_t start, end;

	string str = "f(x, y, z) = 3(sin x)(-5x^2 - y^2 + 3z) + (x^3 - xyz)^3 - 20z";
	// string str = "f(x, y, z) = 3x + yz";
	// string str = "f(x, y, z) = 0 + 3x + 0";

	cout << endl << "Function to be parsed: " << str << endl << endl;

	start = clock();
	functor <double> f(str);
	end = clock();

	cout << endl << "Construction: " << (end - start) /
		(double) CLOCKS_PER_SEC << " seconds " << endl;

	cout << string(100, '-') << endl;
	f.print();
	cout << string(100, '-') << endl;

	cout << "f(2, 3, 4) = " << f(2, 3, 4) << endl;

	cout << endl << f.display() << endl; */

	table <double> tbl {
		variable <double> {"e", exp(1)},
		variable <double> {"pi", acos(-1)},
		variable <double> {"phi", (1 + sqrt(5))/2},
		functor <double> {"g(x, y) = x^3 + y^2"}
	};

	functor <double> h("h(x) = 2^(x^2)", tbl);

	functor <double> g("m(x) = sin x", tbl);

	cout << endl << h.display() << endl;
	
	cout << string(100, '-') << endl;
	h.print();
	cout << string(100, '-') << endl;
	cout << "h(4) = " << h(4) << endl;
	cout << string(100, '-') << endl;

	functor <double> dh_dx = h.differentiate("x");

	cout << dh_dx.display() << endl;
	
	cout << string(100, '-') << endl;
	dh_dx.print();
	cout << string(100, '-') << endl;

	cout << "h'(4) = " << dh_dx(4) << endl;

	functor <double> a = h + g;
	functor <double> b = h - g;
	functor <double> c = h * g;
	functor <double> d = h / g;

	cout << endl << a.display() << endl;
	cout << b.display() << endl;
	cout << c.display() << endl;
	cout << d.display() << endl;

	functor <double> p("f(s_1, s_2) = s_1*s_2");

	cout << endl << "p: " << p << endl;
	p.print();

	cout << "p(4, 5): " << p(4, 5) << endl;

	functor <double> smp("f(x) = sum^{x}_{i = 0} (sin(i) * i^2)");

	cout << endl << "smp: " << smp << endl;
	smp.print();

	cout << endl << "smp(3) = " << smp(3) << endl;
	cout << "smp(4) = " << smp(4) << endl;
	cout << "smp(10) = " << smp(10) << endl;
}

void test_matrix()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING MATRIX TEST" << endl;
	cout << string('#', 100) << endl;

	/* double **mat = new double *[2];
	mat[0] = {1, 2};
	mat[1] = {0, 1}; */

	matrix <double> m(2, 2, 8);

	cout << endl << "m:" << endl << std::string(30, '=')
		<< endl << m.display() << endl;
	
	m.set(0, 1, 5);

	cout << endl << "m.get(0, 1) = " << m.get(0, 1) << endl;

	cout << endl << "m:" << endl << std::string(30, '=')
		<< endl << m.display() << endl;
	
	matrix <double> a({{1, 2}, {0, 4}, {5, 6}});

	cout << endl << "a:" << endl << std::string(30, '=')
		<< endl << a.display() << endl;
	
	a[0][1] = 7;

	cout << endl << "a[0][1] = " << a[0][1] << endl;

	cout << endl << "a:" << endl << std::string(30, '=')
		<< endl << a.display() << endl;

	matrix <double> b(4, 6, [](size_t i, size_t j) {return i * j;});
	
	cout << endl << "b:" << endl << std::string(30, '=')
		<< endl << b.display() << endl;
	
	matrix <double> c(3, 3, [](size_t i, size_t j) {return (double) i / (j + 1);});
	matrix <double> d(3, 3, [](size_t i, size_t j) {return (double) i * j;});

	cout << endl << "Testing addition and subtraction constructors..." << endl;

	cout << endl << "c:" << endl << std::string(30, '=')
		<< endl << c.display() << endl;
	
	cout << endl << "d:" << endl << std::string(30, '=')
		<< endl << d.display() << endl;

	matrix <double> e(3, 3, [&](size_t i, size_t j) {return c[i][j] + d[i][j];});
	matrix <double> f(3, 3, [&](size_t i, size_t j) {return c[i][j] - d[i][j];});
	
	cout << endl << "e:" << endl << std::string(30, '=')
		<< endl << e.display() << endl;
	
	cout << endl << "f:" << endl << std::string(30, '=')
		<< endl << f.display() << endl;
	
	matrix <double> g(3, 3, [](size_t i, size_t j) {return pow(i, j);});
	matrix <double> h(3, 3, [](size_t i, size_t j) {return pow(i, 1.0/(j + 1));});

	cout << endl << "Testing addition and subtraction with operators..." << endl;

	cout << endl << "g:" << endl << std::string(30, '=')
		<< endl << g.display() << endl;
	
	cout << endl << "h:" << endl << std::string(30, '=')
		<< endl << h.display() << endl;

	matrix <double> i = g + h;
	matrix <double> j = g - h;
	
	cout << endl << "i:" << endl << std::string(30, '=')
		<< endl << i.display() << endl;
	
	cout << endl << "j:" << endl << std::string(30, '=')
		<< endl << j.display() << endl;
	
	matrix <double> A({{1, 0.5}, {6.6, 1}});
	matrix <double> B({{1, 4}, {0, 3}});

	cout << endl << "Testing multiplication with operator..." << endl;
	
	cout << endl << "A:" << endl << std::string(30, '=')
		<< endl << A.display() << endl;
	
	cout << endl << "B:" << endl << std::string(30, '=')
		<< endl << B.display() << endl;

	matrix <double> C = A * B;
	
	cout << endl << "C:" << endl << std::string(30, '=')
		<< endl << C.display() << endl;

	cout << endl << "Testing slice feature..." << endl;

	matrix <double> T({
			{1, 2, 3, 4},
			{4, 5, 6, 1},
			{4, 3, 1, 3},
			{4, 2, 1, 2},
			{0, 1, 8, 3}
	});
	
	cout << endl << "T:" << endl << std::string(30, '=')
		<< endl << T.display() << endl;

	matrix <double> S = T.slice({1, 0}, {3, 0});

	cout << endl << "S:" << endl << std::string(30, '=')
		<< endl << S.display() << endl;

	cout << endl << "Testing determinant calculation..." << endl;

	matrix <double> Q({
			{1, 2, 7},
			{4, 5, 8},
			{6, 7, 9}
	});

	cout << endl << "Q:" << endl << std::string(30, '=')
		<< endl << Q.display() << endl;

	cout << endl << "Q.determinant() [" << Q.determinant() << "]" << endl;
	
	cout << endl << "Testing matrix transposition..." << endl;

	A = std::vector <std::vector <double>> {
		{1, 2},
		{3, 4},
		{5, 6}
	};
	
	cout << endl << "A:" << endl << std::string(30, '=')
		<< endl << A.display() << endl;

	T = A.transpose();

	cout << endl << "T:" << endl << std::string(30, '=')
		<< endl << T.display() << endl;
	
	cout << endl << "Testing minors and cofactors..." << endl;

	A = std::vector <std::vector <double>> {
		{1, 4, 7},
		{3, 0, 5},
		{-1, 9, 11}
	};

	cout << endl << "A:" << endl << std::string(30, '=')
		<< endl << A.display() << endl;

	cout << endl << "M(1, 2) [" << A.minor(1, 2) << "]" << endl;
	cout << endl << "C(1, 2) [" << A.cofactor(1, 2) << "]" << endl;

	
	cout << endl << "Testing cofactor matrix and adjugate matrix features..." << endl;

	A = std::vector <std::vector <double>> {
		{-3, 2, -5},
		{-1, 0, -2},
		{3, -4, 1}
	};

	cout << endl << "A:" << endl << std::string(30, '=')
		<< endl << A.display() << endl;

	C = A.cofactor();
	
	cout << endl << "C:" << endl << std::string(30, '=')
		<< endl << C.display() << endl;
	
	matrix <double> Adj = A.adjugate();
	
	cout << endl << "Adj:" << endl << std::string(30, '=')
		<< endl << Adj.display() << endl;
	
	cout << endl << "Testing inverse matrix..." << endl;

	A = std::vector <std::vector <double>> {
		{-3, 2, -5},
		{-1, 0, -2},
		{3, -4, 1}
	};

	cout << endl << "A:" << endl << std::string(30, '=')
		<< endl << A.display() << endl;

	matrix <double> Inv = A.inverse();
	
	cout << endl << "Inv:" << endl << std::string(30, '=')
		<< endl << Inv.display() << endl;
}

void test_root_finding()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING ROOT FINDING TEST" << endl;
	cout << string('#', 100) << endl;

	string str = "f(x) = (x^2 - 3)(x - 4)(x + 5)";
	
	cout << endl << "Function: " << str << endl;

	functor <double> f(str);

	vector <double> visited;

	double epsilon = 1e-9;

	double x_next;
	double x_prev;
	double x_0;

	x_0 = -1;

	cout << endl << "Running tests with epsilon as " << epsilon << endl;

	cout << endl << "Initial guess: " << x_0 << endl;

	functor <double> df_dx = f.differentiate("x");

	cout << endl << "Derivative: " << df_dx.display() << endl;

	double delta = 0.15;
	double factor = 1;

	int iters = 10;
	int rounds = 10;

	cout << endl << "Running for " << iters
		<< " iterations [Newton-Raphson]" << endl;

	x_prev = x_0;
	for (int i = 1; i <= iters; i++) {
		x_next = x_prev - f(x_prev)/df_dx(x_prev);
		x_prev = x_next;

		cout << "\tIteration #" << i << ": x_next [" << x_next << "]" << endl;
	}
	
	cout << endl << "Running for " << iters
		<< " iterations [Modified Newton-Raphson]" << endl;

	x_prev = x_0;
	for (int i = 1; i <= iters; i++) {
		x_next = x_prev - f(x_prev)/df_dx(x_0);
		x_prev = x_next;

		cout << "\tIteration #" << i << ": x_next [" << x_next << "]" << endl;
	}
	
	auto check = [&](double in) {
		for (double val : visited) {
			if (fabs(in - val) <= epsilon)
				return true;
		}

		return false;
	};


	cout << endl << "Running for " << rounds << " rounds on "
		<< iters << " iterations each and " << delta << " as delta"
		<< " [Newton-Raphson]" << endl;
	
	for (int i = 1; i <= rounds; i++) {
		cout << "\tRound #" << i << endl;

		x_prev = x_0;

		for (int j = 1; j <= iters; j++) {
			// cout << "\t\tIteration #" << j << ": x_prev [" << x_prev << "]" << endl;
			
			if (find(visited.begin(), visited.end(), x_prev) != visited.end()) {
				cout << "\t\tROUND FAILURE, hit [" << x_prev << "] again" << endl;
				break;
			}
			
			visited.push_back(x_prev);

			x_next = x_prev - f(x_prev)/df_dx(x_prev);
			x_prev = x_next;
		
			cout << "\t\tIteration #" << i << ": x_next [" << x_next << "]" << endl;

			/* cout << "\t\t\tVisited:" << endl;
			for (double val : visited)
				cout << "\t\t\t\t" << val << endl; */
		}
		
		x_0 += (delta * factor);

		cout << "\t\t\tShifting initial guess [x_0] to " << x_0 << endl;

		delta *= -1;
		factor++;
	}
}

void test_multivariable_root_finding()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING MULTIVARIABLE ROOT FINDING TEST" << endl;
	cout << string('#', 100) << endl;

	string str = "h(x, y) = x^2 + 2y - 5";

	cout << endl << "Function: " << str << endl;

	functor <double> h(str);

	pair <double, double> h_0 = {-1, 5};
	
	cout << endl << "Initial guess: (" << h_0.first << ", " << h_0.second << ")" << endl;

	vector <string> vars = {"x", "y"};

	cout << endl << h << endl;

	matrix <functor <double>> gradient(2, 1, [&](size_t i, size_t j) {
		return h.differentiate(vars[i]);
	});

	cout << endl << gradient << endl;
	
	matrix <functor <double>> hessian(2, 2, [&](size_t i, size_t j) {
		return h.differentiate(vars[i]).differentiate(vars[j]);
	});

	cout << endl << hessian << endl;
}

void test_gmp()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING GMP LIBRARY TEST" << endl;
	cout << string('#', 100) << endl;

	mpz_class a, b, c;

	a = "402398470187412908740294780214732984701298471032984701234873289";
	b = "-543554947962576597265298562765483756";

	c = a * b;

	cout << endl << "c: " << c << endl;
	cout << "Absolute value: " << abs(c) << endl;

	string input;

	// cout << endl << "Enter any number: ";
	// cin >> input;

	mpz_class x;

	// gmp_sscanf(input.c_str(), "%Zd", &x);

	// cout << endl << "x: " << x << endl;

	mpf_class f("432.42342342342342342342343442", 100);
	mpf_class h("5478943.2345890434324242342432", 100);

	cout << "f: " << f << endl;
	cout << "h: " << h << endl;

	int n = 100;
	gmp_printf("fixed point mpf %.*Ff with %d digits\n", n, h, n);
	
	mpf_class d = f / h;
	gmp_printf("fixed point mpf %.*Ff with %d digits\n", n, d, n);

	mpf_class l = trunc(d);
	gmp_printf("fixed point mpf %.*Ff with %d digits\n", n, l, n);

	mpf_class g("545689890890.000000000000000", 100);
	gmp_printf("fixed point mpf %.*Ff with %d digits\n", n, g, n);

	if (cmp(g, trunc(g)) == 0)
		cout << "g is an integer" << endl;
}

void test_node()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING STREE AND NODE TEST" << endl;
	cout << string('#', 100) << endl;

	// node <double> nd = ("4 + 654 - 231");
	stree st("45 + 65");

	st.print(1, 0);

	node <double> nd("43 + 65645");

	nd.print();
}

void test_table()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING OF TABLE TEST" << endl;
	cout << string('#', 100) << endl;

	table <double> tbl;
	
	vector <variable <double>> vals {
		variable <double> {"x", 12.0},
		variable <double> {"ran", true, 123},
		variable <double> {"y", 13423.423},
		variable <double> {"this", true, 12.0}
	};

	cout << endl << "Inserting variables...\n" << endl;
	for (variable <double> v : vals) {
		cout << "\tInserting: " << v << endl;
		tbl.insert_var(v);
	}

	cout << "After Populating:" << endl;
	tbl.print_var();

	cout << "Testing Find Variable:" << endl;

	variable <double> temp;
	for (variable <double> v : vals) {
		cout << endl << "Trying to find " << v << endl;
		temp = tbl.find_var(v.symbol());
		
		cout << "Returned " << temp << endl;
		tbl.print();
	}
	
	vector <std::string> funcs {
		"f(x) = x^4 + 6",
		"h(x) = 232x^7 - 90",
		"g(x, y) = x^2 + y^2"
	};

	for (std::string str : funcs) {
		cout << "\tInserting: " << str << endl;
		tbl.insert_ftr(functor <double> (str));
	}

	cout << "After Populating:" << endl;
	tbl.print_ftr();

	cout << "Testing Find Function:" << endl;

	vector <std::string> fnames {
		"f",
		"g",
		"h"
	};

	functor <double> tmp("");
	for (std::string str : fnames) {
		cout << endl << "Trying to find " << str << endl;
		tmp = tbl.find_ftr(str.substr());
		
		cout << "Returned " << tmp << endl;
		tbl.print_ftr();
	}

	tbl.print();
}

void test_config()
{
	cout << string('#', 100) << endl;
	cout << endl << "BEGINNING CONFIG TEST" << endl;
	cout << string('#', 100) << endl;
	
	vector <opcode> codes {
		op_add,
		op_sub,
		op_mul,
		op_div,
		op_exp,
		op_sin,
		op_cos,
		op_tan,
		op_csc,
		op_sec,
		op_cot,
		op_log
	};

	config <double> cfg;

	cout << "size of cfg: " << sizeof(cfg) << endl;

	token *t;
	for (opcode i : codes) {
		t = cfg.alloc_opn(i);
		cout << "TOKEN: " << endl;
		cout << "\t" << t->str() << endl;
	}

	node <double> *nd;
	for (opcode i : codes) {
		nd = new node <double> {cfg.alloc_opn(i), {}, &cfg};
		cout << "NODE: " << endl;
		// cout << "\t" << t->str() << endl;
		nd->print();
	}

	cout << "============[POST]============" << endl;
	nd = new node <double> {cfg.alloc_opn("sin"), {}, &cfg};
	cout << "NODE: " << endl;
	nd->print();
}

void test_gram_shmidt()
{
	cout << string('#', 100) << endl;
	cout << "BEGINNING GRAM-SHMIDT TEST" << endl;
	cout << string('#', 100) << endl;

	using ld = vector <double>;
	
	vector <element <double>> span {
		ld {1, 1, 1},
		ld {1, 0, 1},
		ld {3, 2, 3}
	};

	cout << endl << "Constructing an orthogonal basis for"
		<< " the linear space spanned by the following vectors." << endl;

	for (auto m : span)
		cout << m << endl;

	/* auto cross = [](const matrix <double> &a, const matrix <double> &b) {
		double acc = 0;
		for (size_t i = 0; i < a.get_cols(); i++)
			acc += a[0][i] * b[0][i];
		return acc;
	}; */

	/* matrix <double> nmat;
	for (size_t i = 1; i < span.size(); i++) {
		nmat = span[i];
		for (size_t j = 0; j < i; j++)
			nmat = nmat - (cross(span[i], basis[j]) / cross(basis[j], basis[j])) * basis[j];
		basis.push_back(nmat);
	} */

	/* element <double> nelem;
	for (size_t i = 1; i < span.size(); i++) {
		nelem = span[i];

		for (size_t j = 0; j < i; j++)
			nelem = nelem - (inner(span[i], basis[j]) / inner(basis[j], basis[j])) * basis[j];
		basis.push_back(nelem);
	} */
	
	vector <element <double>> basis;

	basis = utility::gram_shmidt(span);

	cout << endl << "Resulting orthogonal basis:" << endl;

	for (auto m : basis)
		cout << m << endl;

	span = {
		ld {0, 2, 1, 0},
		ld {1, -1, 0, 0},
		ld {1, 2, 0, -1},
		ld {1, 0, 0, 1}
	};

	// basis = {span[0].normalize()};

	cout << endl << "Constructing an orthonormal basis for"
		<< " the linear space spanned by the following vectors." << endl;
	
	/* for (size_t i = 1; i < span.size(); i++) {
		nelem = span[i];

		for (size_t j = 0; j < i; j++)
			nelem = nelem - (inner(span[i], basis[j]) / inner(basis[j], basis[j])) * basis[j];

		basis.push_back(nelem.normalize());
	} */
	
	basis = utility::gram_shmidt(span);

	cout << endl << "Resulting orthonormal basis:" << endl;

	for (auto m : basis)
		cout << m << endl;
}

void test_ml()
{
	cout << endl << string('=', 100) << endl;
	cout << "BEGINNING ML TEST" << endl;
	cout << string('=', 100) << endl;

	table <double> tbl;

	tbl.insert_var(variable <double> ("e", exp(1)));

	functor <double> steer("steer(sone, stwo, wone, wtwo) = sone * wone + stwo * wtwo");
	functor <double> cost("cost(real, sone, stwo, wone, wtwo) = (real - (sone * wone + stwo * wtwo))^2");
	functor <double> sigma("sigma(x, alpha, minr, maxr) = (maxr - minr)/(1 + e^(-x/alpha)) - (maxr - minr)/2", tbl);

	functor <double> cost_one = cost.differentiate("wone");
	functor <double> cost_two = cost.differentiate("wtwo");

	cout << endl << cost << endl;
	cout << cost_one << endl;
	cout << cost_two << endl;

	int iters = 10;

	cout << endl << "Performing " << iters << " iterations..." << endl;

	double r;
	double l;
	double s;
	while (iters--) {
		r = 10.0 * rand() / RAND_MAX;
		l = 10.0 * rand() / RAND_MAX;

		cout << "r: " << r << endl;
		cout << "l: " << l << endl;

		cout << "\tsteer: " << (s = steer(r, l, 5, 6)) << endl;
		cout << "\tsigma: " << (s = sigma(s, 400, -30, 30)) << endl;
		cout << "\tcost[L]: " << cost(-30, s, 400, -30, 30) << endl;
		// cout << "\t\tcost[L]: " << cost_one(-30, s, 400, -30, 30) << endl;
		cout << "\tcost[R]: " << cost(30, s, 400, -30, 30) << endl;
		// cout << "\t\tcost[R]: " << cost_two(-30, s, 400, -30, 30) << endl;
	}

	functor <double> ft("f(a, b, c, d, l, r, LR, RL) = a * (c*l + d*r) + b * (c*RL + d*LR)");

	cout << endl << "FT: " << ft << endl;
	ft.print();

	functor <double> ft_a = ft.differentiate("a");
	functor <double> ft_c = ft.differentiate("b");
	functor <double> ft_b = ft.differentiate("c");
	functor <double> ft_d = ft.differentiate("d");

	cout << endl << "ft_a: " << ft_a << endl;
	cout << "ft_b: " << ft_b << endl;
	cout << "ft_c: " << ft_c << endl;
	cout << "ft_d: " << ft_d << endl;

	tbl.insert_ftr(ft);

	functor <double> sigmoid("s(x, alpha) = alpha/(1 + e^(-x)) - alpha/2", tbl);
	
	cout << endl << "SIGMOID: " << sigmoid << endl;
	sigmoid.print();

	functor <double> sig_a = sigmoid.differentiate("alpha");
	functor <double> sig_x = sigmoid.differentiate("x");

	cout << endl << "sig_a: " << sig_a << endl;
	cout << endl << "sig_x: " << sig_x << endl;

	tbl.insert_ftr(sigmoid);

	cout << endl;

	tbl.print();

	functor <double> compound("steer(a, b, c, d, A, r, l, LR, RL) = s(f(a, b, c, d, r, l, LR, RL), A)", tbl);

	cout << endl << "COMPOUND: " << compound << endl;
	compound.print();

	functor <double> comp_a = compound.differentiate("a");
	functor <double> comp_b = compound.differentiate("b");
	functor <double> comp_c = compound.differentiate("c");
	functor <double> comp_d = compound.differentiate("d");

	cout << endl << "comp_a: " << comp_a << endl;
	cout << endl << "comp_b: " << comp_b << endl;
	cout << endl << "comp_c: " << comp_c << endl;
	cout << endl << "comp_d: " << comp_d << endl;
}

void test_lagrange_interpolation()
{
	cout << string(100, '#') << endl;
	cout << "BEGINNING LAGRANGE INTERPOLATION TEST" << endl;
	cout << string(100, '#') << endl;

	vector <pair <double, double>> data {
		{1, 6},
		{2, -7},
		{3, 6},
		{7, 1}
	};

	functor <double> ftr = utility::interpolate_lagrange(data);

	cout << endl << "Function (Curve): " << ftr << endl;
}

void test_general()
{
	cout << string(100, '=') << endl;
	cout << "GENERAL TEST" << endl;
	cout << string(100, '=') << endl;

	cout << endl << string(30, '-') << endl;
	cout << "Factorials" << endl;
	cout << string(30, '-') << endl;

	functor <double> f("f(x) = x! + 5!");

	cout << endl << f << endl;
	f.print();

	cout << "f(5) = " << f(5) << endl;

	cout << endl << string(30, '-') << endl;
	cout << "LU Factorization" << endl;
	cout << string(30, '-') << endl;

	matrix <double> A  = vector <vector <double>> {
		{2, -1, 2},
		{-4, 6, 3},
		{-4, -2, 8}
	};

	cout << endl << "A:" << endl << A;

	pair <matrix <double>, matrix <double>> out = utility::lu_factorize(A);

	matrix <double> L = out.first;
	matrix <double> U = out.second;
	
	cout << endl << "L:" << endl << L;
	cout << endl << "U:" << endl << U;
}

vector <function <void ()>> tests = {
	test_general
};

int main()
{
	// yydebug = 1;

	for (auto t : tests) {
		try {
			t();
		} catch (node <double> ::undefined_symbol e) {
			cout << "ERROR:\t" << e.what() << endl;
		}
	}
}
