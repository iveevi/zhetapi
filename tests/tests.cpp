// Engine headers
#include <node_manager.hpp>
#include <barn.hpp>
#include <function.hpp>

#define TEST3

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
		zhetapi::node_manager <double, int> tmp(str, barn);

		zhetapi::token *tptr = tmp.value();

		cout << endl << "Value: " << tptr->str() << " (" << tptr << ")" << endl;
	}

#elif defined(TEST3)

	Function <double, int> f = std::string("f(x, y) = x^2 + y");

#endif

	return 0;
}
