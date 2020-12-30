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
	
	model = zhetapi::ml::NeuralNetwork <double> ({
                {8, new zhetapi::ml::Linear <double> ()},
                {10, new zhetapi::ml::Sigmoid <double> ()},
                {10, new zhetapi::ml::ReLU <double> ()},
                {9, new zhetapi::ml::Linear <double> ()}
        }, initializer);
}
