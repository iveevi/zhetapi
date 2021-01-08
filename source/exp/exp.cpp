// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>
#include <iomanip>

// Engine headers
#include <matrix.hpp>

using namespace std;
using namespace zhetapi;

#define MIN_SIZE	10
#define MAX_SIZE	1000
#define STEP		10

int main()
{
	clock_t start;
	clock_t end;
	
	ofstream fout("data/mat_mult.dat");

	Matrix <double> c;
	for (size_t i = MIN_SIZE; i <= MAX_SIZE; i += STEP) {
		Matrix <double> a(i, i,
			[](size_t r, size_t c) {
				return rand()/((double) RAND_MAX);
			}
		);
		
		Matrix <double> b(i, i,
			[](size_t r, size_t c) {
				return rand()/((double) RAND_MAX);
			}
		);

		start = clock();

		a * b;

		end = clock();

		fout << i << "\t";
		fout << setprecision(6) << fixed
			<< (end - start)/((double) CLOCKS_PER_SEC) << endl;
	}
}
