// C/C++ headers
#include <ios>

// Engine headers
#include <node_manager.hpp>
#include <barn.hpp>
#include <function.hpp>
#include <engine.hpp>

#define TEST4

using namespace std;
using namespace zhetapi;

int main()
{

#if defined(TEST1)
	
	Barn <double, int> barn;

	Variable <double> var {"e", 2.17};

	barn.put(var);

	cout << "var: " << var << endl;
	cout << "barn-var: " << barn.retrieve <double> ("e") << endl;
	cout << "barn-var: " << barn.get("e")->str() << endl;

#elif defined(TEST2)
	
	Barn <double, int> barn;

	barn.put(Variable <double> {"e", exp(1)});
	barn.put(Variable <double> {"pi", acos(-1)});

	std::string str;

	cout << endl << "Beginning Tests..." << endl;

	while (getline(cin, str)) {
		try {
			zhetapi::node_manager <double, int> tmp(str, barn);

			zhetapi::token *tptr = tmp.value();

			cout << endl << "Value: " << tptr->str() << " (" << tptr << ")" << endl;
		} catch (Barn <double, int> ::unknown_operation_overload_exception e) {
			cout << e.what() << endl;
		} catch (zhetapi::node_manager <double, int> ::error e) {
			cout << e.what() << endl;
		} catch (...) {
			cout << "Another Exception" << endl;
		}
	}

#elif defined(TEST3)

	Function <double, int> f = std::string("f(x, y) = x^2 + y");

	try {
		cout << "f(10, -4) = " << f(10, -4)->str() << std::endl;

		cout << "df/dx (10, -4) = " << f.derivative("x", 10, -4)->str() << std::endl;
	} catch (Barn <double, int> ::unknown_operation_overload_exception e) {
		cout << e.what() << endl;
	}

#elif defined(TEST4)

	Engine <double, int> eng("src/std_simplifications.ztp");

	node_manager<double, int> mg("x^2 * 1", {"x"});

	mg.print();

	mg.simplify(eng);

#endif

	return 0;
}
