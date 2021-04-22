#include "global.hpp"

typedef void (*exporter)(Engine *);

static int assess_library(string);

int compile_library(vector <string> files, string output)
{
	string sources = "";
	for (string file : files)
		sources += file + " ";

	// Assume the file ends with ".cpp" for now
	string outlib = output;
	if (output.empty()) {
		string base = files[0].substr(0, files[0].length() - 4);

		outlib = base + ".zhplib";
	}

	string opts = " --no-gnu-unique -g -rdynamic -fPIC -shared ";
	string idir = " -I engine ";
	string ldir = " -L$PWD/bin ";
	string libs = " -lzhp ";
	string rpath = " -Wl,-rpath $PWD/bin ";

	string cmd = "g++-8 " + opts + idir + sources + ldir + libs + " -o " + outlib + rpath;

	if (verbose)
		cout << cmd << endl;

	if (system(cmd.c_str())) {
		printf("Fatal error: could not compile library \'%s\'\n", outlib.c_str());
		
                return -1;
        }

	return 0;
}

int assess_libraries(vector <string> files)
{
	for (string file : files)
		assess_library(file);

	return 0;
}

static int assess_library(string file)
{
	Engine tmp;

	const char *dlsymerr = nullptr;

	// Load the library
	void *handle = dlopen(file.c_str(), RTLD_NOW);

	// Check for errors
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: unable to open file '%s': %s\n", file.c_str(), dlsymerr);

		return -1;
	}

	// Get the exporter
	void *ptr = dlsym(handle, "zhetapi_export_symbols");

	// Check for errors
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: could not find \"zhetapi_export_symbols\" in file '%s': %s\n", file.c_str(), dlsymerr);

		return -1;
	}

	exporter exprt = (exporter) ptr;

	if (!exprt) {
		printf("Failed to extract exporter\n");

		return -1;
	}

	exprt(&tmp);

	tmp.list_registered(file);

	return 0;
}

int import_library(string file)
{
	const char *dlsymerr = nullptr;

	// Load the library
	void *handle = dlopen(file.c_str(), RTLD_NOW);

	// Check for errors
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: unable to open file '%s': %s\n", file.c_str(), dlsymerr);

		return -1;
	}

	// Get the exporter
	void *ptr = dlsym(handle, "zhetapi_export_symbols");

	// Check for errors
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: could not find \"zhetapi_export_symbols\" in file '%s': %s\n", file.c_str(), dlsymerr);

		return -1;
	}

	exporter exprt = (exporter) ptr;

	if (!exprt) {
		printf("Failed to extract exporter\n");

		return -1;
	}

	exprt(engine);

	return 0;
}
