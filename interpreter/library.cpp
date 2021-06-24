#include "global.hpp"
#include <dlfcn.h>

static int assess_library(string file)
{
	Module tmp("md-tmp");

	const char *dlsymerr = nullptr;

	// Load the library
	void *handle = dlopen(file.c_str(), RTLD_NOW);

	// Check for errors
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: unable to open file '%s': %s\n", file.c_str(), dlsymerr);

		return -1;
	}

	// Display the linted version
	void *ptr1 = dlsym(handle, "__zhp_linted_version__");
	
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: could not find \"__zhp_linted_version__\" in file '%s': %s\n", file.c_str(), dlsymerr);

		return -1;
	}

	const char *(*lver)() = (const char *(*)()) ptr1;

	if (!lver) {
		printf("Failed to extract version linter\n");

		return -1;
	}

	printf("Linted version = <%s>\n", lver());

	if (strcmp(__get_linted_version__(), lver()) == 0)
		printf("\tSame as interpreter version\n");
	else
		printf("\tWARNING: Differs from interpreter's version, <%s>\n", __get_linted_version__());

	// Get the exporter (TODO: add __ to the name)
	void *ptr2 = dlsym(handle, "zhetapi_export_symbols");

	// Check for errors
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: could not find \"zhetapi_export_symbols\" in file '%s': %s\n", file.c_str(), dlsymerr);

		return -1;
	}

	Module::Exporter exporter = (Module::Exporter) ptr2;

	if (!exporter) {
		printf("Failed to extract exporter\n");

		return -1;
	}

	exporter(&tmp);

	tmp.list_attributes(cout);

	dlclose(handle);

	// TODO: Handle errors

	return 0;
}

int assess_libraries(vector <string> files)
{
	for (string file : files)
		assess_library(file);

	return 0;
}

// TODO: add gcc compilation flags
int compile_library(vector <string> files, string output)
{
	// Assuming zhetapi is already installed
	string sources = "/usr/local/include/zhetapi/version.cpp ";
	for (string file : files)
		sources += file + " ";

	// Assume the file ends with ".cpp" for now
	string outlib = output;
	if (output.empty()) {
		string base = files[0].substr(0, files[0].length() - 4);

		outlib = base + ".zhplib";
	}

	// should also optimize
	string opts = " --no-gnu-unique -g -rdynamic -fPIC -shared -D__ZHETAPI_LINTED_VERSION__=\'\""
		+ std::string(__get_linted_version__()) + "\"\'";

	// TOOD: remove idir
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
