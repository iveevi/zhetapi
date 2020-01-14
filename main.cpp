// C++ Standard Libraries
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

// Custom Built Libraries
#include "all.h"

// Include Directives
using namespace std;
using namespace tokens;
using namespace trees;
using namespace chrono;

// Benchmarking Tools
#define INPUT "0.123 ^ 12 - 456 * 2 ^ 32 / 89"

high_resolution_clock::time_point start_t, end_t;
duration <double, micro> time_span;

int main()
{
	start_t = high_resolution_clock::now();
        
	token_tree <double> tr(INPUT);
	tr.print();
	dp_ptr(tr.value()->dptr);
	
	end_t = high_resolution_clock::now();
	
	time_span = duration_cast <duration <double, micro>> (end_t - start_t);
	cout << "Process elapsed " << time_span.count()
		<< " microseconds." << endl;
}
