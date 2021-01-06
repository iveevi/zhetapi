#include "global.hpp"

int compile_library(string file)
{
	// Assume the file ends with ".cpp" for now
	string base = file.substr(0, file.length() - 4);
	string outlib = base + ".zhplib";
	string outso = base + ".so";

	printf("Compiling '%s' into zhp library '%s'...\n", file.c_str(), outso.c_str());
	string cmd = "g++-8 --no-gnu-unique -I engine -I inc/hidden -I inc/std \
				-g -rdynamic -fPIC -shared " + file  + " -o " + outso;

	int ret = system(cmd.c_str());

	ifstream fin(file);
	ofstream fout(outlib);

	fout << file << endl;

	string str;
	while (fin >> str) {
		if (str.substr(0, 16) == "ZHETAPI_REGISTER") {
			int i = 17;
			string ident;

			while (i < str.length() && str[i] != ')') {
				ident += str[i];

				i++;
			}

			fout << ident << endl;
		}
	}

	return ret;
}
