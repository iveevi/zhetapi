#include "port.hpp"

bool function_computation()
{
	using namespace zhetapi;

	bench tb;

	const int iters = 1;

	Function f = "f(x, y) = x^2 + 2x + ln(x^3 * y - 36)";

	tb = bench();
	for (int i = 0; i < iters; i++)
		f(5, 4);
	
	cout << "Time for regular computation: " << tb << endl;

	f.set_threads(8);

	tb = bench();
	for (int i = 0; i < iters; i++)
		f(5, 4);
	
	cout << "Time for multi-threaded computation: " << tb << endl;

	return true;
}

bool function_compilation_testing()
{
	using namespace zhetapi;

#ifdef ZHP_FUNCTION_COMPILE_GENERAL
	// Timers
	bench tb;

	// Verdict status
	int count = 0;

	// The function to be compiled
	Function f = "f(x, y, z) = x^2 + 2x + x^2 + 24yz - z^5";

	// Create the typedef for the compiled function
	typedef Token *(*ftr1)(Token *, Token *, Token *);

	tb = bench();
	ftr1 tmp1 = (ftr1) f.compile_general();

	cout << "Compilation time for first: " << tb << endl;

	// Allocate operands
	Operand <double> *op1 = new Operand <double> (3.544);
	Operand <int> *op2 = new Operand <int> (56);
	Operand <double> *op3 = new Operand <double> (343.54454);

	// Compare outputs
	Token *t1 = f(3.544, 56, 343.54454);
	Token *t2 = tmp1(op1, op2, op3);

	cout << "t1: " << t1->str() << endl;
	cout << "t2: " << t2->str() << endl;

	if (t1 == t2) {
		cout << "\nEqual tokens!" << endl;

		count++;
	}

	// Repeat, with a different function
	Function g = "g(x) = sin(x) - log(x)";

	// Create the typedef for the compiled function
	typedef Token *(*ftr2)(Token *);

	tb = bench();
	ftr2 tmp2 = (ftr2) g.compile_general();

	cout << "Compilation time for second: " << tb << endl;

	// Allocate operands
	Operand <double> *op4 = new Operand <double> (4.767);

	// Compare outputs
	Token *t3 = g(4.767);
	Token *t4 = tmp2(op4);

	cout << "t1: " << t3->str() << endl;
	cout << "t2: " << t4->str() << endl;

	if (t3 == t4) {
		cout << "\nEqual tokens!" << endl;

		count++;
	}
	
	// Free resources
	delete op1;
	delete op2;
	delete op3;
	delete op4;

	// Deliver the verdict
	cout << "Verdict: " << count << "/2 matches." << endl;

	return (count == 2);
#endif
	cout << "Cannot conduct proper testing of this feature." << endl;

	return false;
}
