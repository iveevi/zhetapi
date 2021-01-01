// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <std/activation_classes.hpp>
#include <std/optimizer_classes.hpp>
#include <network.hpp>

#include <dataset.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	auto initializer = []() {
                return 0.5 - (rand()/(double) RAND_MAX);
        };

	ml::NeuralNetwork <double> model;
	
	model = ml::NeuralNetwork <double> ({
                {11, new zhetapi::ml::Linear <double> ()},
                {10, new zhetapi::ml::Sigmoid <double> ()},
                {10, new zhetapi::ml::ReLU <double> ()},
                {10, new zhetapi::ml::Sigmoid <double> ()},
                {9, new zhetapi::ml::Linear <double> ()}
        }, initializer);

	model.randomize();

	Vector <double> in(8, [](size_t i) {return rand()/((double) RAND_MAX);});
	Vector <double> out(9, [](size_t i) {return rand()/((double) RAND_MAX);});

	ml::Optimizer <double> *opt = new ml::MeanSquaredError <double> ();

	model.set_cost(opt);

	DataSet <double> ins;
	DataSet <double> outs;
	for (size_t i = 0; i < 5000; i++) {
		auto rng = [](size_t i) {
			return rand()/((double) RAND_MAX);
		};

		ins.push_back(Vector <double> (11, rng));
		outs.push_back(Vector <double> (9, rng));
	}

        std::chrono::high_resolution_clock clk;
	
	std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end;

	double t_norm;
	for (size_t i = 0; i < 10; i++) {
		start = clk.now();

		for (size_t i = 0; i < ins.size(); i++)
			model.compute_no_cache(ins[i]);

		end = clk.now();

		t_norm = std::chrono::duration_cast
			<std::chrono::microseconds> (end - start).count();
		
		start = clk.now();

		cout << "t_norm = " << t_norm << endl;
	}
}
