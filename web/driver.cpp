#include <assert.h>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "expression.h"
#include "table.h"
#include "variable.h"

#define MAX_COUNT	4096

#define FIFO_R_FILE	"/home/ram/zhetapi/build/driver_in"
#define FIFO_W_FILE	"/home/ram/zhetapi/build/driver_out"

using namespace std;

/* Structures associated with state of application */
table <double> tbl;

vector <string> history;

enum code {
	ASSIGN_VAR,
	ASSIGN_FTR,
	EXPRESSION
};

struct smnt {
	code cd;
	string str;
};

string pref;

vector <string> scan(string line)
{
	vector <string> strs;

	string acc;

	bool paren = false;
	for (int i = 0; i < line.length(); i++) {
		char c = line[i];

		switch (c) {
		case ',':
			if (!paren) {
				strs.push_back(acc);
				acc.clear();
			} else {
				acc += c;
			}

			break;
		case '(':
			paren = true;
			acc += c;
			break;
		case ')':
			paren = false;
			acc += c;
			break;
		default:
			acc += c;
			break;
		}
	}

	if (!acc.empty())
		strs.push_back(acc);

	vector <string> out;

	pref = "";
	for (string str : strs) {
		int i = str.find("=");

		if (i != string::npos) {
			string left = str.substr(0, i);
			string right = str.substr(i + 1);

			left = std::regex_replace(left, std::regex("^ +| +$|( ) +"), "$1");
			right = std::regex_replace(right, std::regex("^ +| +$|( ) +"), "$1");

			i = left.find('(');

			string name;
			if (i != string::npos) {
				str = std::regex_replace(str, std::regex("^ +| +$|( ) +"), "$1");

				name = left.substr(0, i);

				Function <double> nf(str, tbl);
				
				tbl.remove_ftr(name);
				tbl.insert_ftr(nf);

				cout << "nf([n]), [n] = " << nf.ins() << endl;

				if (nf.ins() == 1) {
					std::time_t rawtime;
					std::tm* timeinfo;
					
					char buffer[80];

					std::time(&rawtime);
					timeinfo = std::localtime(&rawtime);

					std::strftime(buffer,80,"%Y-%m-%d-%H-%M-%S",timeinfo);

					cout << "SINGLE VARIABLE" << endl;
					cout << buffer << endl;
					pref = "GRAPH[" + str + "]: " + string(buffer);
					// system(string("echo -e \"" + str + "\\n10\n0\" | /home/ram/zhetapi/build/graph > /home/ram/zhetapi/web/data/graph_" + buffer + "_10_0").c_str());
				}

				continue;
			}

			double val = expression <double> ::in_place_evaluate(right, tbl);

			try {
				tbl.get_var(left).set(val);
			} catch (...) {
				variable <double> vr(left, val);
				tbl.insert_var(vr);
			}
		} else {
			double val;
		
			int vars = 0;
			string name;

			table <double> cpy = tbl;

			int ses = 2;
			while (true) {
				try {
					val = expression <double> ::in_place_evaluate(str, cpy);
					cout << "Attempt successful" << endl;
					break;
				} catch (node <double> ::undefined_symbol e) {
					cout << "Is function: \"" << e.what() << "\"" << endl;
					cpy.insert_var(variable <double> {e.what(), 0.0});
					name = e.what();
					vars++;
				}
			}
			
			cout << "vars: " << vars << endl;

			if (vars == 1) {
				string ftr = "f(" + name + ") = " + str;

				std::time_t rawtime;
				std::tm* timeinfo;
				
				char buffer[80];

				std::time(&rawtime);
				timeinfo = std::localtime(&rawtime);

				std::strftime(buffer,80,"%Y-%m-%d-%H-%M-%S",timeinfo);

				cout << "SINGLE VARIABLE" << endl;
				cout << buffer << endl;
				pref = "GRAPH[" + ftr + "]: " + string(buffer);
			} else {
				out.push_back(str);
			}
		}
	}

	return out;
}

static void parent(void)
{
	uint8_t data[MAX_COUNT];
	uint8_t *in;
	size_t count;
	int fout;
	int fin;
	int n;

	fin = open(FIFO_R_FILE, O_RDONLY);

	assert(fin > 0);

	string out;
	string stripped;

	do {
		n = read(fin, &count, 4);
		
		assert(n == 4);
		assert(count < MAX_COUNT);

		n = read(fin, data, count);
		assert(n == count);
		data[count] = 0;

		fout = open(FIFO_W_FILE, O_WRONLY);
		assert(fout > 0);
		
		stripped = (char *) data;

		vector <string> sts = scan(stripped);

		if (sts.empty()) {
			if (!pref.empty())
				out = pref;
			else
				out = "";
		} else {
			try {
				out = pref + to_string(expression <double> ::in_place_evaluate(sts[0], tbl));
			} catch(node <double> ::undefined_symbol e) {
				out = "Could not identify symbol or variable $" +
					e.what() + "$ [Undefined Symbol Error].";
			} catch (...) {
				out = "Could not evaluate expression [Unkown Error].";
			}
		}
		
		n = write(fout, out.c_str(), out.length());

		close(fout);
	} while (count > 0);

	close(fin);
}

int main(int argc, char *argv[])
{
	vector <variable <double>> vars {
		variable <double> {"pi", acos(-1)},
		variable <double> {"e", exp(1)}
	};
	
	for (variable <double> vr : vars)
		tbl.insert_var(vr);
	
	mknod(FIFO_R_FILE, S_IFIFO | 0666, 0);
	mknod(FIFO_W_FILE, S_IFIFO | 0666, 0);

	try {
		parent();
	} catch (node <double> ::undefined_symbol e) {
		cout << "e.what: " << e.what() << endl;
	}

	assert(!remove(FIFO_R_FILE));
	assert(!remove(FIFO_W_FILE));

	return 0;
}
