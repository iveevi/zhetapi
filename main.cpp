// C++ Standard Libraries
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <unistd.h>

// Custom Built Libraries
#include "all.h"

// Include Directives
using namespace std;
using namespace tokens;
using namespace trees;
using namespace chrono;

// Benchmarking Tools
#define INPUT "(log 2 8) * sin (3.1415926535 / 2) + tan(3.141526535 / 4)"

// Default # of tests to 1
#ifndef TESTS

#define TESTS 1

#endif

#define LINE 75

struct bench {
	duration <double, micro> time;
	double mem;
	double rss;
	double vm;
};

high_resolution_clock::time_point start_t, end_t;

void mem_usage(double &vm_t, double &rss_t)
{
	// vm_usage = 0.0;
	// resident_set = 0.0;
	ifstream fin("/proc/self/stat", ios_base::in);

	string pid, comm, state, ppid, pgrp, session, tty_nr;
	string tpgid, flags, minflt, cminflt, majflt, cmajflt;
	string utime, stime, cutime, cstime, priority, nice;
	string O, rval, start_time;

	unsigned long vsize;
	long rss;

	fin >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
		>> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
		>> utime >> stime >> cutime >> cstime >> priority >> nice
		>> O >> rval >> start_time >> vsize >> rss;
	fin.close();

	vm_t = vsize / 1024.0;
	rss_t = rss * sysconf(_SC_PAGE_SIZE) / 1024;
}

// Benchmark function
bench bench_mark(int num)
{
	cout << "Beginning Benchmark Test:" << endl << endl;
	cout << "Input: " << INPUT << endl << endl;

	start_t = high_resolution_clock::now();
	bench out;

	token_tree <double> tr(INPUT);

	cout << string(LINE, '-') << endl;
	cout << "TEST #" << num << " - ";
	
	tr.print();
	dp_var(tr.value());
	out.mem += sizeof(tr);
	
	mem_usage(out.vm, out.rss);
	end_t = high_resolution_clock::now();
	out.time = duration_cast <duration <double, micro>> (end_t - start_t);

	return out;
}

int main()
{
	// Perform mutliple benchmarks
	bench total {duration <double, micro> (), 0, 0};
	bench mark;

	for (int i = 0; i < TESTS; i++) {
		mark = bench_mark(i + 1);

		total.time += mark.time;
		total.rss += mark.rss;
		total.vm += mark.vm;
		total.mem += mark.mem;
	}

	cout << string(LINE, '-') << endl;
	total.time /= TESTS;
	// total.rss /= TESTS;
	// total.mem /= TESTS;
	// total.vm /= TESTS;

	cout << endl << "Average program (process) statistics: " << TESTS << " tests performed " << endl;
	cout << "\tProcess elapsed " << total.time.count()
		<< " microseconds on average." << endl;
	cout << "\tToken_tree tr consumed " << total.mem << " bytes" << endl;
	cout << "\tVirtual Memory: " << total.vm << " kilobytes" << endl;
	cout << "\tResident set size: " << total.rss << " kilobytes" << endl;
	cout << "\tToken_tree tr consumed " << total.mem / TESTS << " bytes on average" << endl;
	cout << "\tVirtual Memory: " << total.vm / TESTS << " kilobytes on average" << endl;
	cout << "\tResident set size: " << total.rss / TESTS << " kilobytes on average" << endl << endl;

	cout << "Library class statistics" << endl;
	cout << "\tToken: " << sizeof(token) << " bytes" << endl;
	cout << "\tOperand: " << sizeof(operand <double>) << " bytes" << endl;
	cout << "\tOperation: " << sizeof(operation <operand <double>>) << " bytes" << endl;
	cout << "\tParser: " << sizeof(parser <double>) << " bytes" << endl;
	cout << "\tDefaults: " << sizeof(defaults <double>) << " bytes" << endl;
	cout << "\tVariable: " << sizeof(variable <double>) << " bytes" << endl;
	cout << "\tToken_tree: " << sizeof(token_tree <double>) << " bytes" << endl;
}