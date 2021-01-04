// Source headers
#include "global.hpp"

// Zhetapi API storage
zhetapi::Barn <double, int> barn;


// Constants
zhetapi::Operand <bool> *op_true = new zhetapi::Operand <bool> (true);
zhetapi::Operand <bool> *op_false = new zhetapi::Operand <bool> (false);

size_t line = 1;

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

	barn.put(zhetapi::Variable <double, int> (op_true->copy(), "true"));
	barn.put(zhetapi::Variable <double, int> (op_false->copy(), "false"));

	return parse();
}
