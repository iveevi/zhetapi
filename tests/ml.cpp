#include <cstdlib>
#include <iostream>
#include <random>

#include <activations.hpp>
#include <matrix.hpp>
#include <network.hpp>

using namespace std;
using namespace ml;

int main()
{
	ReLU <double> act;

	cout << act(10) << endl;

	Matrix <double> A(3, 3);

	cout << A << endl;

	srand(clock());

	A.randomize([]() {return rand()/(double) RAND_MAX;});

	cout << A << endl;

	DeepNeuralNetwork <double> model({
		{4, ReLU <double> ()},
		{10, ReLU <double> ()},
		{10, ReLU <double> ()},
		{1, ReLU <double> ()}
	}, []() {return rand()/(double) RAND_MAX;});

	cout << model({1, 1, 1, 1}) << endl;

	model.randomize();

	cout << model({1, 1, 1, 1}) << endl;

	model.randomize();

	cout << model({1, 1, 1, 1}) << endl;
}