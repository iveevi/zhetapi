#include <iostream>
#include <fstream>

#include <std/activation_classes.hpp>
#include <network.hpp>

using namespace std;

ifstream images("train-images-idx3-ubyte", ios::binary);
ifstream labels("train-labels-idx1-ubyte", ios::binary);

zhetapi::ml::DeepNeuralNetwork <double> model({
	{784, new zhetapi::ml::ReLU <double> ()},
	{20, new zhetapi::ml::ReLU <double> ()},
	{20, new zhetapi::ml::ReLU <double> ()},
	{10, new zhetapi::ml::Sigmoid <double> ()}
}, []() {return 0.5 - (rand()/(double) RAND_MAX);});

int reverse(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

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
	for(size_t i = 0; i < 60000; i++) {
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
		cout << "Item #" << (i + 1) << " -- Supposed to be " << (int) actual << ", got " << model(in) << endl;
	}
}

int main()
{
	model.randomize();

	read_mnist();
}
