#include <matrix.hpp>
#include <vector.hpp>

#include <std/interval.hpp>

#include <image.hpp>

#include <chrono>
#include <string>
#include <iostream>

using namespace std;
using namespace zhetapi;
using namespace zhetapi::utility;

// Typedefs
using tclk = chrono::high_resolution_clock;
using tpoint = chrono::high_resolution_clock::time_point;

tclk clk;
tpoint start_t;
tpoint end_t;

int main()
{
	/*
	auto dtime = []() -> ostream & {
		double mcs = chrono::duration_cast
			<chrono::microseconds>
			(end_t - start_t).count();

		return cout << mcs;
	};

	Matrix <double> A(10, 10, runit());
	Matrix <double> B(10, 10, runit());
	Matrix <double> C(10, 10, runit());
	Matrix <double> D;

	double ka = 10 * runit();
	double kb = 10 * runit();

	start_t = clk.now();
	for(size_t i = 0; i < 100; i++) {
		D = ka * A * B + kb * C;
	}
	end_t = clk.now();

	dtime() << "\t";

	start_t = clk.now();
	for(size_t i = 0; i < 100; i++) {
		D = fmak(A, B, C, ka, kb);
	}
	end_t = clk.now();

	dtime() << endl; */
}
