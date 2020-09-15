#include <cstdlib>
#include <iostream>
#include <random>

#include <activation.hpp>
#include <optimizer.hpp>

#include <std_activation_classes.hpp>
#include <std_optimizer_classes.hpp>

#include <matrix.hpp>
#include <vector.hpp>
#include <network.hpp>
#include <tensor.hpp>

using namespace std;
using namespace ml;

int main()
{
	srand(clock());

	DeepNeuralNetwork <double> model({
		{4, new Linear <double> ()},
		{10, new Linear <double> ()},
		{10, new Linear <double> ()},
		{2, new Linear <double> ()}
	}, []() {return 0.5 - rand()/(double) RAND_MAX;});

	cout << "1:\t" << model({1, -1, 5, -2}) << endl;

	model.randomize();

	cout << "2:\t" << model({1, 1, 1, 1}) << endl;

	model.randomize();

	cout << "3:\t" << model({1, 1, 1, 1}) << endl;

	model.learn({1, 1, 1, 1}, {2, 2}, new MeanSquaredError <double> ());
}
