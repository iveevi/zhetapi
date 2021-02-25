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

	DNN <> net(4, {
		Layer <> (6, new ReLU <double> ()),
		Layer <> (5, new ReLU <double> ()),
		Layer <> (6, new ReLU <double> ()),
		Layer <> (7, new ReLU <double> ())
	});

	Optimizer <double> *opt = new Adam <double> ();
	Erf <double> *cost = new MeanSquaredError <double> ();

	net.set_optimizer(opt);
	net.set_cost(cost);

	vector <Vector <double>> ins;
	vector <Vector <double>> outs;

	for (size_t i = 0; i < 100; i++) {
		Vector <double> in(4, gen);
		Vector <double> out(7, gen);

		ins.push_back(in);
		outs.push_back(out);
	}

	start_t = clk.now();
	
	for (size_t i = 0; i < 10; i++)
		train_dataset_perf(net, ins, outs, 20, cost, 0, 1);
	
	end_t = clk.now();

	double mcs = chrono::duration_cast
		<chrono::microseconds>
		(end_t - start_t).count();
	
	cout << "total time = " << mcs << endl;

	delete opt;
	delete cost;
}
