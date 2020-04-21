#include <iostream>
#include <fstream>

#include <signal.h>

#include "expression.h"
#include "var_stack.h"

using namespace std;

/* Shared files */
ofstream fout("../build/driver.out");
ifstream fin("../build/driver.in");

int parent;

/* Structures with state of application */
vector <string> history;

var_stack <double> vst;

void strip();

void notify(int i)
{
	signal(SIGHUP, notify);
	printf("[Driver]: Received SIGHUP signal, proceeding with appropriate action...\n");
	strip();
	goto loop;
}

void strip()
{
	string line;
	
	getline(fin, line);

	printf("[Driver]: Read line \"%s\"\n", line.c_str());

	// Ignore header number for now
	string input;

	istringstream iss(line);
	int holder;
	char c;

	while(iss >> c) {
		if (c == '#') {
			iss >> holder;
			iss.get(); // should be tab

			iss >> input;
		}
	}

	cout << "[Driver]: Extracted input \"" << input << "\"" << endl;
	cout << "Sending SIGINT to process @ " << parent << endl;
	kill(parent, SIGINT);
}

void cache_pid()
{
	ifstream pid_file("../build/pid");

	// add label/header if really necessary
	// ie. if there are multiple asynchronous
	// processes running while the server is
	pid_file >> parent;
}

void load_constants()
{
	// implement later
}

void load_defaults()
{
	// implement later
}

int main(int argc, char *argv[])
{
	// setup
	cache_pid();


	vector <variable <double>> vals {
		variable <double> {"pi", acos(-1)},
		variable <double> {"e", exp(1)}
	};
	
	for (variable <double> v : vals)
		vst.insert(v);

	signal(SIGHUP, notify);

	/* try {
		//fout << "\\text{" << line << "}" << endl;
		//fout << "\\[" << expression <double> ::in_place_evaluate(line) << "\\]" << endl;
		fout << expression <double> ::in_place_evaluate(line, vst) << endl;
	} catch(node <double> ::undefined_symbol e) {
		fout << "\\text{Could not identify symbol or variable} $"
			<< e.what() << "$ \\text{ [Undefined Symbol Error].}" << endl;
	} catch (...) {
		fout << "\\text{Could not evaluate expression [Unkown Error].}" << endl;
	} */

loop:
	while (true);
}
