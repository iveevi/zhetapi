// Source headers
#include "global.hpp"

// Execution modes
enum mode {
	interpret,	// Interpreting files
	build,		// Compiling libraries
	unbox,		// Show symbols
};

// Interpretation kernel
static int interpreter(string infile)
{
	if (!freopen(infile.c_str(), "r", stdin)) {
		printf("Fatal error: failed to open file '%s'.\n", infile.c_str());

		exit(-1);
	}

	file = infile;
	
	// Register builtin symbols
	barn.put(Registrable("print", &print));
	barn.put(Registrable("println", &println));

	barn.put(Variable <double, int> (op_true->copy(), "true"));
	barn.put(Variable <double, int> (op_false->copy(), "false"));

	return parse();
}

// Main
int main(int argc, char *argv[])
{
	int ret;

	char c;

	vector <string> sources;

	string infile;
	string output;

	char *next;
	int index;

	mode md = interpret;
	while ((c = getopt(argc, argv, ":c:d:o:")) != EOF) {
		switch (c) {
		case 'c':
			md = build;

			index = optind - 1;

			sources.clear();
			while (index < argc) {
				next = strdup(argv[index++]);

				if (next[0] == '-')
					break;
			
				sources.push_back(next);
			}
			
			break;
		case 'd':
			md = unbox;

			index = optind - 1;

			sources.clear();
			while (index < argc) {
				next = strdup(argv[index++]);

				if (next[0] == '-')
					break;
				
				sources.push_back(next);
			}
			
			break;
		case 'o':
			output = optarg;
			break;
		default:
			break;
		}
	}

	if (optind < argc)
		infile = argv[optind];

	switch (md) {
	case build:
		ret = compile_library(sources, output);
		break;
	case unbox:
		ret = assess_libraries(sources);
		break;
	case interpret:
		ret = interpreter(infile);
		break;
	}

	return ret;
}
