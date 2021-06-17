// Source headers
#include "global.hpp"

// Execution modes
enum mode {
	interpret,	// Interpreting files
	build,		// Compiling libraries
	unbox,		// Show symbols
	help		// Help guide
};

// Setup the default include directories
bool verbose = false;

Args idirs {
	".",
	"/usr/local/lib/zhp"
};

// Display guide
static int guide()
{
	printf("Usage: zhetapi [options] file...\n");
	printf("Options:\n");
	printf(" -c\t\tCompiles the files into a single library.\n");
	printf(" -d\t\tDisplays exported symbols in the libraries specified.\n");
	printf(" -h\t\tDisplay the guide for the interpreter.\n");
	printf(" -o <file>\tPlace the compiled library into <file>.\n");
	printf(" -L <directory>\tAdd <directory> to the interpreter's "
		"library search paths.\n");

	return 0;
}

// Source file directory
static inline string __get_dir(string file)
{
	size_t i = file.length() - 1;
	while (file[i--] != '/');

	return file.substr(0, i + 1);
}

// Inline kernel
static int inline_interpret()
{
	Engine *ctx = new Engine(true);

	cout << "Running Zhetapi version <" << __get_linted_version__() << ">" << endl;
	while (true) {
		cout << "> ";

		string str;
		getline(cin, str);

		if (!cin) {
			cout << endl;

			exit(0);
		}

		if (str.empty())
			continue;

		StringFeeder sf('\n' + str);
		parse_global(&sf, ctx, idirs, __get_linted_version__(), true);
	}

	delete ctx;
}

// Interpretation kernel
static int interpreter(string infile)
{
	if (infile.empty())
		inline_interpret();

	if (!freopen(infile.c_str(), "r", stdin)) {
		printf("Fatal error: failed to open file '%s'.\n", infile.c_str());

		exit(-1);
	}

	Engine *ctx = new Engine(true);
	
	StringFeeder sf = file_feeder(infile);
	int ret = parse_global(&sf, ctx, idirs, __get_linted_version__());
	
	delete ctx;

	return ret;
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
	while ((c = getopt(argc, argv, ":c:d:o:L:hv")) != EOF) {
		switch (c) {
		case 'c':
			if (md != help)
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
			if (md != help)
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
		case 'L':
			idirs.push_back(optarg);

			break;
		case 'h':
			md = help;

			break;
		case 'v':
			verbose = true;

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
		idirs.insert(std::next(idirs.begin()), __get_dir(infile));

		ret = interpreter(infile);
		break;
	case help:
		ret = guide();
		break;
	}

	return ret;
}
