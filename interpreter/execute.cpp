#include "global.hpp"

// Zhetapi API storage
Engine *engine = new Engine(true);

Token *execute(string str)
{
	// Skip comments
	if (str[0] == '#')
		return nullptr;
	
	// cout << "line = " << line << ", STR = " << str << endl;

	vector <string> tmp = split(str);
	
	size_t tsize = tmp.size();

	if (tsize > 1) {
		zhetapi::Token *tptr = nullptr;
		
		node_manager::undefined_symbol us("");
		bool pe = false;

		try {
			zhetapi::node_manager mg(engine, tmp[tsize - 1]);

			tptr = mg.value(engine);
		} catch (const Engine::unknown_op_overload &e)  {
			cout << "err: " << e.what() << endl;
			exit(-1);
		} catch (const node_manager::undefined_symbol &e) {
			us = e;
			pe = true;
		}

		for (int i = tsize - 2; i >= 0; i--) {
			string ftr = tmp[i] + " = " + tmp[tsize - 1];

			try {
				zhetapi::Function f(ftr);

				engine->put(f);
			} catch (...) {
				if (pe) {
					symbol_error_msg(us.what(), engine);
					exit(-1);
				}

				engine->put(tmp[i], tptr);
			}
		}
		
		delete tptr;
	} else {		
		// All functions and algorithms are stored in engine
		node_manager mg;
	
		/* cout << "BARN:" << endl;
		engine.print(); */

		Token *tptr = nullptr;

		try {
			mg = node_manager(engine, str);
			
			tptr = mg.value(engine);
		} catch (const Engine::unknown_op_overload &e)  {
			cout << "err evaluating \'" << str << "\'\n" << e.what() << endl;
			exit(-1);
		} catch (const node_manager::undefined_symbol &e) {
			symbol_error_msg(e.what(), engine);

			exit(-1);
		} catch (const Token::unknown_attribute &e) {
			cout << "Error at line " << line
				<< ": " << e.what() << endl;
			
			exit(-1);
		}

		/* cout << "mg:" << endl;
		mg.print(); */

		// "Execute" the statement
		return tptr;
	}

	return nullptr;
}
