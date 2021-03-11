#include "global.hpp"

// Zhetapi API storage
Barn barn;

Token *execute(string str)
{
	// Skip comments
	if (str[0] == '#')
		return nullptr;

	vector <string> tmp = split(str);
	
	size_t tsize = tmp.size();

	if (tsize > 1) {
		zhetapi::Token *tptr = nullptr;
		
		node_manager::undefined_symbol us("");
		bool pe = false;

		try {
			zhetapi::node_manager mg(tmp[tsize - 1], &barn);

			tptr = mg.value();
		} catch (const Barn::unknown_operation_overload &e)  {
			cout << "err: " << e.what() << endl;
			exit(-1);
		} catch (const node_manager::undefined_symbol &e) {
			us = e;
			pe = true;
		}

		for (int i = tsize - 2; i >= 0; i--) {
			string ftr = tmp[i] + " = " + tmp[tsize - 1];

			try {
				zhetapi::Function f(ftr, &barn);

				barn.put(f);
			} catch (...) {
				if (pe) {
					cout << "err:" << us.what() << endl;
					exit(-1);
				}

				barn.put(tptr, tmp[i]);
			}
		}
		
		delete tptr;
	} else {		
		// All functions and algorithms are stored in barn
		node_manager mg;
	
		/* cout << "BARN:" << endl;
		barn.print(); */

		Token *tptr = nullptr;

		try {
			mg = node_manager(str, &barn);
			
			tptr = mg.value();
		} catch (const Barn::unknown_operation_overload &e)  {
			cout << "err evaluating \'" << str << "\'" << e.what() << endl;
			exit(-1);
		} catch (const node_manager::undefined_symbol &e) {
			cout << "Error at line " << line
				<< ": undefined symbol \""
				<< e.what() << "\"" << endl;

			exit(-1);
		}

		/* cout << "mg:" << endl;
		mg.print(); */

		// "Execute" the statement
		return tptr;
	}

	return nullptr;
}
