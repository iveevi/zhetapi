// C/C++ headers
#include <ctime>
#include <cmath>
#include <iostream>

#include <dlfcn.h>

// Engine headers
#include <function.hpp>

#define TEST4

using namespace std;

#if defined(TEST1)

	const size_t rounds = 1000000;

	time_t startt;
	time_t endt;

	zhetapi::Barn <double, int> barn;

	zhetapi::token *__fx_general(double x)
	{
		zhetapi::token *i1 = new zhetapi::operand <double> (x);

		zhetapi::token *c1 = new zhetapi::operand <double> (54);

		zhetapi::token *int1 = barn.compute("*", {i1, c1});
		zhetapi::token *int2 = barn.compute("ln", {i1});

		return barn.compute("+", {int1, int2});
	}
	
	zhetapi::token *__fx_general_optimized(double x)
	{
		zhetapi::token *i1 = new zhetapi::operand <double> (x);

		zhetapi::token *c1 = new zhetapi::operand <double> (54);

		zhetapi::token *int1 = barn.compute_optimized("*", {i1, c1});
		zhetapi::token *int2 = barn.compute_optimized("ln", {i1});

		return barn.compute_optimized("+", {int1, int2});
	}

	inline zhetapi::token *__fx_general_inline(double x)
	{
		// Variables
		zhetapi::token *i1 = new zhetapi::operand <double> (x);

		// Constants
		zhetapi::token *c1 = new zhetapi::operand <double> (54);

		return barn.compute("+", {barn.compute("*", {i1, c1}), barn.compute("ln", {i1})});
	}
	
	inline zhetapi::token *__fx_general_inline_optimized(double x)
	{
		// Variables
		zhetapi::token *i1 = new zhetapi::operand <double> (x);

		// Constants
		zhetapi::token *c1 = new zhetapi::operand <double> (54);

		return barn.compute_optimized("+", {barn.compute_optimized("*", {i1, c1}), barn.compute_optimized("ln", {i1})});
	}

	double __fx_in_R_out_R(double x)
	{
		return 54.0 * x + log(x);
	}

	inline double __fx_in_R_out_R_inline(double x)
	{
		return 54.0 * x + log(x);
	}

#endif

int main()
{

#if defined(TEST1)

	zhetapi::Function <double, int> fx = "f(x) = x * 54 + ln(x)";

	/////////////////////////////////////////////////////////////
	cout << "Running with zhetapi::Function:" << endl;

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		fx(10);

	endt = clock();

	cout << "\tTime: " << (endt - startt)/((double) CLOCKS_PER_SEC) << " seconds" << endl;

	/////////////////////////////////////////////////////////////
	cout << "Running with general function:" << endl;

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		__fx_general(10);

	endt = clock();

	double gen = (endt - startt)/((double) CLOCKS_PER_SEC);

	cout << "\tTime: " << gen << " seconds" << endl;

	/////////////////////////////////////////////////////////////
	cout << "Running with inlined general function:" << endl;

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		__fx_general_inline(10);

	endt = clock();

	cout << "\tTime: " << (endt - startt)/((double) CLOCKS_PER_SEC) << " seconds" << endl;
	
	/////////////////////////////////////////////////////////////
	cout << "Running with optimized general function:" << endl;

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		__fx_general_optimized(10);

	endt = clock();

	double opt = (endt - startt)/((double) CLOCKS_PER_SEC);

	cout << "\tTime: " << opt << " seconds" << endl;

	/////////////////////////////////////////////////////////////
	cout << "Running with optimized inlined general function:" << endl;

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		__fx_general_inline_optimized(10);

	endt = clock();

	cout << "\tTime: " << (endt - startt)/((double) CLOCKS_PER_SEC) << " seconds" << endl;

	/////////////////////////////////////////////////////////////
	cout << "Running with specific function:" << endl;

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		__fx_in_R_out_R(10);

	endt = clock();

	cout << "\tTime: " << (endt - startt)/((double) CLOCKS_PER_SEC) << " seconds" << endl;

	/////////////////////////////////////////////////////////////
	cout << "Running with inlined specific function:" << endl;

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		__fx_in_R_out_R_inline(10);

	endt = clock();

	cout << "\tTime: " << (endt - startt)/((double) CLOCKS_PER_SEC) << " seconds" << endl;

#elif defined(TEST2)

	const char file[] = "./__gen_fone.so";

	cout << 24 << endl;

	system("g++ --no-gnu-unique -I engine -I inc/hidden -I inc/std __gen_fone.cpp -rdynamic -shared -fPIC -o __gen_fone.so");

	void *handle = dlopen(file, RTLD_NOW);

	cout << "handle: " << handle << endl;

	const char *dlsym_error = dlerror();

	if (dlsym_error) {
		cerr << "Err: " << dlsym_error << '\n';
		
		dlclose(handle);
		
		return 1;
	}

	typedef zhetapi::token *(*fctn)(zhetapi::token *, zhetapi::token *);

	fctn f = (fctn) dlsym(handle, "__gen_fone");

	void *ptr = dlsym(handle, "__gen_fone");

	cout << "ptr: " << ptr << endl;

	cout << "f: " << f << endl;

	dlsym_error = dlerror();
	
	if (dlsym_error) {
		cerr << "Cannot load symbol 'F': " << dlsym_error << '\n';
		
		dlclose(handle);
		
		return 1;
	}
	
	// dlclose(handle);

	cout << f(new zhetapi::operand <int> (34), new zhetapi::operand <int> (5))->str() << endl;

	system("readelf -Ws __gen_fone.so | grep UNIQUE");

#elif defined(TEST3)

	// Problems with complex numbers
	zhetapi::Function <double, int> fx1 = "fone(x, y) = 34x + y^2";
	zhetapi::Function <double, int> fx2 = "ftwo(x, y) = 34x - 54y + 65x";
	zhetapi::Function <double, int> fx3 = "ftre(x, y) = 34 * sin(x) + 54354 - y";

	// fx1.print();
	// fx2.print();
	// fx3.print();

	void *ptr1 = fx1.compile_general();
	void *ptr2 = fx2.compile_general();
	void *ptr3 = fx3.compile_general();

	cout << "Address of fx1 general: " << ptr1 << endl;
	cout << "Address of fx2 general: " << ptr2 << endl;
	cout << "Address of fx3 general: " << ptr3 << endl;

	typedef zhetapi::token *(*ftr)(zhetapi::token *, zhetapi::token *);

	ftr f1 = (ftr) ptr1;
	ftr f2 = (ftr) ptr2;
	ftr f3 = (ftr) ptr3;

	cout << "Address of fx1 general (cast): " << f1 << endl;
	cout << "Address of fx2 general (cast): " << f2 << endl;
	cout << "Address of fx3 general (cast): " << f3 << endl;

	cout << "f1: " << f1(new zhetapi::operand <int> (3), new zhetapi::operand <int> (4))->str() << endl;
	cout << "fx1: " << fx1(3, 4)->str() << endl;

	cout << "f2: " << f2(new zhetapi::operand <int> (3), new zhetapi::operand <int> (4))->str() << endl;
	cout << "fx2: " << fx2(3, 4)->str() << endl;

	cout << "f3: " << f3(new zhetapi::operand <int> (3), new zhetapi::operand <int> (4))->str() << endl;
	cout << "fx3: " << fx3(3, 4)->str() << endl;

#elif defined(TEST4)

	size_t rounds = 100000;

	clock_t startt;
	clock_t endt;

	zhetapi::Function <double, int> fx = "f(x) = x * 54 + ln(x)";

	/////////////////////////////////////////////////////////////
	cout << "Running with zhetapi::Function:" << endl;

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		fx(10);

	endt = clock();

	double gen = (endt - startt)/((double) CLOCKS_PER_SEC);

	cout << "\tTime: " << gen << " seconds" << endl;

	/////////////////////////////////////////////////////////////

	cout << "\nCompiling general function:" << endl;

	startt = clock();

	typedef zhetapi::token *(*ftr)(zhetapi::token *);
	
	ftr gfx = (ftr) fx.compile_general();

	endt = clock();

	gen = (endt - startt)/((double) CLOCKS_PER_SEC);

	cout << "\tTime: " << gen << " seconds" << endl;

	/////////////////////////////////////////////////////////////
	
	cout << "\nRunning with general function:" << endl;

	zhetapi::token *opd = new zhetapi::operand <int> (10);

	startt = clock();

	for (size_t i = 0; i < rounds; i++)
		gfx(opd);

	endt = clock();

	gen = (endt - startt)/((double) CLOCKS_PER_SEC);

	cout << "\tTime: " << gen << " seconds" << endl;

#endif

}
