// C++ Standard Libraries
#include <iostream>
#include <string>
#include <thread>

// Custom Built Libraries
#include "all.h"

// Include Directives
using namespace std;
using namespace tokens;
using namespace trees;

// Benchmarking Tools
#define INPUT "123 - 456"

clock_t start, end;

int main()
{
	start = clock();
        
	token_tree <double> tr(INPUT);
	tr.print();
	dp_ptr(tr.value()->dptr);
	
	end = clock();
	
	cout << "\nAlgorithm Elapsed Time: " 
		<< (double) (start - end) / (double) CLOCKS_PER_SEC 
		<< " seconds." << endl;
}
