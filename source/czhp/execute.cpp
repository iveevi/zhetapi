#include "global.hpp"

zhetapi::Token *execute(string str)
{
	// "Execute" the statement
	zhetapi::Token *tptr = nullptr;
	
	try {
		zhetapi::node_manager <double, int> mg(str, barn);

		tptr = mg.value();
	} catch (zhetapi::node_manager <double, int> ::undefined_symbol e) {
		cout << "error: " << e.what() << endl;
	}

	return tptr;
}