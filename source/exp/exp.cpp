#include <all/ml.hpp>

#include <std/interval.hpp>

#include <chrono>
#include <string>
#include <iostream>

using namespace std;
using namespace zhetapi;
using namespace zhetapi::ml;
using namespace zhetapi::utility;

// Typedefs
using tclk = chrono::high_resolution_clock;
using tpoint = chrono::high_resolution_clock::time_point;

tclk clk;
tpoint start_t;
tpoint end_t;

int main()
{
	Interval <> unit = 10.0_I;

	auto gen = [&](size_t i) -> double {
		return unit.uniform();
	};

	DNN <> net(40, {
		Layer <> (60, new ReLU <double> ()),
		Layer <> (50, new ReLU <double> ()),
		Layer <> (60, new ReLU <double> ()),
		Layer <> (70, new ReLU <double> ())
	});

	Optimizer <double> *opt = new Adam <double> ();
	Erf <double> *cost = new MeanSquaredError <double> ();

	net.set_optimizer(opt);
	net.set_cost(cost);

	vector <Vector <double>> ins;
	vector <Vector <double>> outs;

	for (size_t i = 0; i < 1000; i++) {
		Vector <double> in(40, gen);
		Vector <double> out(70, gen);

		ins.push_back(in);
		outs.push_back(out);
	}

	double mcs;
	double avg;
	double tot;

	avg = 0;
	tot = 0;
	for (size_t i = 0; i < 10; i++) {
		start_t = clk.now();

		train_dataset_perf(net, ins, outs, 100, cost, 0, 1);
	
		end_t = clk.now();

		mcs = chrono::duration_cast
			<chrono::microseconds>
			(end_t - start_t).count();
		
		avg += mcs;
		tot += mcs;
	}

	avg /= (1000 * 100);
	tot /= 1000;
	
	cout << "total regular time = " << tot << " ms" << endl;
	cout << "\taverage regular time = " << avg << " ms" << endl;

	// Switch to optimized
	fopt = true;

	avg = 0;
	tot = 0;
	for (size_t i = 0; i < 10; i++) {
		start_t = clk.now();

		train_dataset_perf(net, ins, outs, 100, cost, 0, 1);
	
		end_t = clk.now();

		mcs = chrono::duration_cast
			<chrono::microseconds>
			(end_t - start_t).count();
		
		avg += mcs;
		tot += mcs;
	}

	avg /= (1000 * 100);
	tot /= 1000;
	
	cout << "total optimized time = " << tot << " ms" << endl;
	cout << "\taverage optimized time = " << avg << " ms" << endl;

	delete opt;
	delete cost;
}
