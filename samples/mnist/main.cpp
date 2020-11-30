// C/C++ headers
#include <iostream>
#include <fstream>
#include <vector>

// Engine headers
#include <std/activation_classes.hpp>
#include <network.hpp>

#define IMAGES	60000
#define SIZE	28

using namespace std;

ifstream images("train-images-idx3-ubyte", ios::binary);
ifstream labels("train-labels-idx1-ubyte", ios::binary);

zhetapi::ml::DeepNeuralNetwork <double> model({
	{784, new zhetapi::ml::ReLU <double> ()},
	{20, new zhetapi::ml::Sigmoid <double> ()},
	{20, new zhetapi::ml::ReLU <double> ()},
	{10, new zhetapi::ml::Softmax <double> ()}
}, []() {return 0.5 - (rand()/(double) RAND_MAX);});

void read_mnist()
{
	unsigned char tmp;

	images.read((char*)&tmp,sizeof(tmp));
	labels.read((char*)&tmp,sizeof(tmp));

	images.read((char*)&tmp,sizeof(tmp));
	labels.read((char*)&tmp,sizeof(tmp));

	images.read((char*)&tmp,sizeof(tmp));
	images.read((char*)&tmp,sizeof(tmp));
	
	labels.read((char*)&tmp,sizeof(tmp));
	labels.read((char*)&tmp,sizeof(tmp));

	int size = 28 * 28;
	for(size_t i = 0; i < 10; i++) {
		vector <double> pixels;

		for (int k = 0; k < size; k++) {
			unsigned char temp=0;
			images.read((char*)&temp,sizeof(temp));

			pixels.push_back(temp);
		}

		unsigned char actual;

		labels.read((char *) &actual, sizeof(actual));

		zhetapi::Vector <double> in = pixels;

		cout << "==================================================" << endl;

		zhetapi::Vector <double> out = model(in);

		int mi = 0;
		for (int i = 1; i < 10; i++) {
			if (out[mi] < out[i])
				mi = i;
		}

		cout << "Item #" << (i + 1) << " -- Supposed to be " << (int) actual << ", got " << mi + 1 << endl;
	}
}

int main()
{
	model.randomize();

	read_mnist();
}
