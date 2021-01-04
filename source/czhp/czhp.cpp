// Source headers
#include "global.hpp"

// Zhetapi API storage
Barn <double, int> barn;

// Constants
Operand <bool> *op_true = new Operand <bool> (true);
Operand <bool> *op_false = new Operand <bool> (false);

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
	barn.put(Registrable("print", &print));
	barn.put(Registrable("println", &println));

	barn.put(Variable <double, int> (op_true->copy(), "true"));
	barn.put(Variable <double, int> (op_false->copy(), "false"));

	return parse();
}
