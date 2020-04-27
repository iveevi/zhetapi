#include <assert.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "expression.h"
#include "var_stack.h"

#define MAX_COUNT	4096
#define FIFO_R_FILE	"/home/ram/zhetapi/build/web2drv"
#define FIFO_W_FILE	"/home/ram/zhetapi/build/drv2web"

using namespace std;

/* Structures with state of application */
vector <string> history;

var_stack <double> vst;

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

		printf("calculating...\n");

		try {
			out = to_string(expression <double> ::in_place_evaluate(stripped, vst));
		} catch(node <double> ::undefined_symbol e) {
			out = "\\text{Could not identify symbol or variable} $" +
				e.what() + "$ \\text{ [Undefined Symbol Error].}";
		} catch (...) {
			out = "\\text{Could not evaluate expression [Unkown Error].}";
		}

		n = write(fout, out.c_str(), out.length());

		close(fout);
	} while (count > 0);

	close(fin);
}

int main(int argc, char *argv[])
{
	vector <variable <double>> vals {
		variable <double> {"pi", acos(-1)},
		variable <double> {"e", exp(1)}
	};
	
	for (variable <double> v : vals)
		vst.insert(v);
	
	mknod(FIFO_R_FILE, S_IFIFO | 0666, 0);
	mknod(FIFO_W_FILE, S_IFIFO | 0666, 0);

	parent();

	assert(!remove(FIFO_R_FILE));
	assert(!remove(FIFO_W_FILE));

	return 0;
}
