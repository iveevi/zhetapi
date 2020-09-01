#include <cstdlib>
#include <iostream>
#include <random>

#include <activation.hpp>

#include <std_activation_classes.hpp>

#include <matrix.hpp>
#include <vector.hpp>
#include <network.hpp>

using namespace std;
using namespace ml;

int main()
{
	srand(clock());

	DeepNeuralNetwork <double> model({
		{4, new Linear <double> ()},
		{10, new Linear <double> ()},
		{10, new Linear <double> ()},
		{1, new Linear <double> ()}
	}, []() {return 0.5 - rand()/(double) RAND_MAX;});

	cout << model({1, -1, 5, -2}) << endl;

	model.randomize();

	cout << model({1, 1, 1, 1}) << endl;

	model.randomize();

	cout << model({1, 1, 1, 1}) << endl;
}
