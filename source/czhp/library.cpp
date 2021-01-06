#include "global.hpp"

typedef void (*exporter)(Barn <double, int> &);

int compile_library(string file)
{
	// Assume the file ends with ".cpp" for now
	string base = file.substr(0, file.length() - 4);
	string outlib = base + ".zhplib";

	printf("Compiling '%s' into zhp library '%s'...\n", file.c_str(), outlib.c_str());
	string cmd = "g++-8 --no-gnu-unique -I engine -I inc/hidden -I inc/std \
				-g -rdynamic -fPIC -shared " + file  + " source/registration.cpp -o " + outlib;

	return system(cmd.c_str());
}

int assess_library(string file)
{
	Barn <double, int> tmp;

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

	exprt(tmp);

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

	exprt(barn);

	return 0;
}
