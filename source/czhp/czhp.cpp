// Source headers
#include "global.hpp"

// Zhetapi API storage
zhetapi::Barn <double, int> barn;

size_t line = 1;

// Global processing
void process(string statement)
{
	// Skip comments
	if (statement[0] == '#')
		return;

	vector <string> tmp = split(statement);
	
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
		zhetapi::node_manager <double, int> mg(statement, barn);

		// "Execute" the statement
		mg.value();
	}
}

// Main
int main(int argc, char *argv[])
{
	if (argc == 2) {
		if (!freopen(argv[1], "r", stdin)) {
			printf("Fatal error: failed to open file '%s'.\n", argv[1]);

			exit(-1);
		}
	}
	
	// Barn setup	
	barn.put(zhetapi::Registrable("print", &print));
	barn.put(zhetapi::Registrable("println", &println));

	return parse();
}
