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
#define INPUT "0.123 to the power of 18.5645 - (456 * 2 ^ 32 / 89)"

high_resolution_clock::time_point start_t, end_t;
duration <double, micro> time_span;
double vm, rss;

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

int main()
{
	cout << "Beginning Benchmark Test:" << endl << endl;
	cout << "Input: " << INPUT << endl << endl;
	start_t = high_resolution_clock::now();

	token_tree <double> tr(INPUT);
	tr.print();
	dp_ptr(tr.value()->dptr);
	
	mem_usage(vm, rss);
	end_t = high_resolution_clock::now();
	time_span = duration_cast <duration <double, micro>> (end_t - start_t);

	cout << endl << "Program (process) statistics:" << endl;
	cout << "\tProcess elapsed " << time_span.count()
		<< " microseconds." << endl;
	cout << "\tToken_tree tr consumed " << sizeof(tr) << " bytes" << endl;
	cout << "\tVirtual Memory: " << vm << " kilobytes" << endl;
	cout << "\tResident set size: " << rss << " kilobytes" << endl << endl;

	cout << "Library class statistics" << endl;
	cout << "\tToken: " << sizeof(token) << " bytes" << endl;
	cout << "\tOperand: " << sizeof(operand <double>) << " bytes" << endl;
	cout << "\tOperation: " << sizeof(operation <operand <double>>) << " bytes" << endl;
	cout << "\tModule: " << sizeof(module <operand <double>>) << " bytes" << endl;
	cout << "\tVariable: " << sizeof(variable <double>) << " bytes" << endl;
	cout << "\tToken_tree: " << sizeof(token_tree <double>) << " bytes" << endl;
	cout << "\tTtwrapper: " << sizeof(ttwrapper <double>) << " bytes" << endl;
	cout << "\tNode: " << sizeof(node <ttwrapper <double>>) << " bytes" << endl;
	cout << "\tList: " << sizeof(list <node <ttwrapper <double>>>) << " bytes" << endl;
}
