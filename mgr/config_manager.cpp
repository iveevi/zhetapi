// C/C++ headers
#include <fstream>
#include <sstream>
#include <iostream>

// Boost headers
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/range/iterator_range.hpp>

using namespace std;
using namespace boost::filesystem;

// Global variables
vector <path> files;

// Exposers
void expose(string file)
{
	std::ifstream fin(file);

	size_t n = 0;

	cout << endl;

	string line;
	while (getline(fin, line))
		cout << "[" << ++n << "] \"" << line << "\"" << endl;
	
	cout << endl;
}

// Tokensizer
vector <string> tokenize(const string &str)
{
	vector <string> tokens;

	istringstream iss(str);

	string s;
	while (iss >> s)
		tokens.push_back(s);

	return tokens;
}

// Performs commands
void perform(const vector <string> &toks)
{
	// Size
	size_t n = toks.size();

	// Non-empty command
	if (!n)
		return;

	if (toks[0] == "q" || toks[0] == "quit") {
		exit(0);
	} else if (toks[0] == "load") {
		if (n < 2) {
			cout << "\nExpected index to load\n" << endl;
			return;
		}

		size_t index = stoi(toks[1], nullptr, 10);

		if (0 > index || index > files.size()) {
			cout << "\nLoad index out of bounds\n" << endl;
			return;
		}

		expose(files[index - 1].string());
	} else {
		cout << "\nUnknown command \"" << toks[0] << "\"\n" << endl;
	}
}

int main(int argc, char *argv[])
{
	string str = (argc > 1) ? argv[1] : ".";

	cout << "Zhetapi Manager" << endl;
	
	path pwd(str);

	bool config = false;
	if (is_directory(pwd)) {
		for(auto &entry : boost::make_iterator_range(directory_iterator(pwd), {})) {
			if (entry.path().extension() == ".zhpconfig") {
				config = true;

				str = entry.path().string();
			}
		}
	}
		
	if (config) {
		cout << "\nLoading configuration file...\n" << endl;
	} else {
		cout << "\nNo configuration file found. Exiting...\n" << endl;
		return 0;
	}

	vector <string> dirs;

	std::ifstream fin(str);

	string line;
	while (getline(fin, line))
		dirs.push_back(pwd.string() + "/" + line);

	size_t n = 0;
	for (auto pt : dirs) {
		path tmp(pt);
		for(auto &entry : boost::make_iterator_range(directory_iterator(tmp), {})) {
			if (is_regular_file(entry)) {
				cout << "[" << ++n << "] " << entry << endl;
				files.push_back(entry);
			}
		}
	}

	cout << endl;

	// Parsing commands
	vector <string> toks;

	string cmd;
	while (true) {
		cout << "(zhetapi) ";

		getline(cin, cmd);
		
		toks = tokenize(cmd);

		perform(toks);
	}
}