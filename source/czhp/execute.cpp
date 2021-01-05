#include "global.hpp"

Token *execute(string str)
{
	// Skip comments
	if (str[0] == '#')
		return nullptr;

	vector <string> tmp = split(str);
	
	size_t tsize = tmp.size();
	if (tsize > 1) {
		zhetapi::Token *tptr = nullptr;
		
		try {
			zhetapi::node_manager <double, int> mg(tmp[tsize - 1], barn);

			tptr = mg.value();
		} catch (...) {}

		for (int i = tsize - 2; i >= 0; i--) {
			string ftr = tmp[i] + " = " + tmp[tsize - 1];

			try {
				zhetapi::Function <double, int> f = ftr;

				barn.put(f);
			} catch (...) {
				barn.put(tptr, tmp[i]);
			}
		}
		
		delete tptr;
	} else {		
		// All functions and algorithms are stored in barn
		zhetapi::node_manager <double, int> mg(str, barn);

		/* cout << "mg:" << endl;
		mg.print(); */

		// "Execute" the statement
		return mg.value(barn);
	}

	return nullptr;
}
