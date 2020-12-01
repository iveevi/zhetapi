// C/C++ headers
#include <iostream>
#include <fstream>
#include <vector>

// Engine headers
#include <std/activation_classes.hpp>
#include <std/optimizer_classes.hpp>
#include <network.hpp>

#define IMAGES	60000
#define SIZE	28

using namespace std;

// Global variables
ifstream images("train-images-idx3-ubyte", ios::binary);
ifstream labels("train-labels-idx1-ubyte", ios::binary);

zhetapi::ml::NeuralNetwork <double> model({
	{784, new zhetapi::ml::ReLU <double> ()},
	{20, new zhetapi::ml::Sigmoid <double> ()},
	{20, new zhetapi::ml::ReLU <double> ()},
	{10, new zhetapi::ml::Softmax <double> ()}
}, []() {return 0.5 - (rand()/(double) RAND_MAX);});

vector <zhetapi::Vector <double>> imgs;
vector <zhetapi::Vector <double>> exps;

unsigned int tmp;

// Reading images
vector <double> read_image()
{
	vector <double> pixels;

	int size = SIZE * SIZE;
	for (int k = 0; k < size; k++) {
		unsigned char temp=0;
		images.read((char*)&temp,sizeof(temp));

		pixels.push_back(temp);
	}
	
	return pixels;
}

vector <double> read_image_print()
{
	vector <double> pixels;

	int size = SIZE * SIZE;

	cout << "\n " << string(28, '-') << "\n";
	for (int k = 0; k < size; k++) {
		unsigned char temp = 0;

		images.read((char*)&temp,sizeof(temp));

		pixels.push_back(temp);

		if (k % SIZE == 0)
			cout << "|";

		if (temp) {
			cout << "#";
		} else {
			cout << ".";
		}

		if (k % SIZE == 27)
			cout << "|\n";
	}
	cout << " " << string(28, '-') << "\n";
	
	return pixels;
}

// Main function
int main()
{
	// Initialize the model
	srand(clock());

	model.randomize();

	// First 16 bytes
	images.read((char*)&tmp,sizeof(tmp));
	images.read((char*)&tmp,sizeof(tmp));
	images.read((char*)&tmp,sizeof(tmp));
	images.read((char*)&tmp,sizeof(tmp));
	
	// First 8 bytes
	labels.read((char*)&tmp,sizeof(tmp));
	labels.read((char*)&tmp,sizeof(tmp));

	// Pass critique
	auto crit = [](zhetapi::Vector <double> actual, zhetapi::Vector <double> expected) {
		int mi = 0;
		for (int i = 1; i < 10; i++) {
			if (actual[mi] < actual[i])
				mi = i;
		}

		return (expected[mi] == 1);
	};

	cout << boolalpha;

	int size = 28 * 28;
	for(size_t i = 0; i < 20; i++) {
		zhetapi::Vector <double> in = read_image();

		unsigned char actual;

		labels.read((char *) &actual, sizeof(actual));

		// zhetapi::Vector <double> out = model(in);

		/* int mi = 0;
		for (int i = 1; i < 10; i++) {
			if (out[mi] < out[i])
				mi = i;
		} */

		zhetapi::Vector <double> exp(10,
			[&](size_t i) {
				return (i == actual) ? 1.0 : 0.0;
			}
		);

		imgs.push_back(in);
		exps.push_back(exp);

		/* cout << "Item #" << (i + 1) << " -- Supposed to be " << (int) actual << ", got " << mi << endl;
		cout << "\tactual:\t" << out << endl;
		cout << "\texpected:\t" << exp << endl;
		cout << "\tmatch:\t" << crit(out, exp) << endl; */
	}

	zhetapi::ml::Optimizer <double> *opt = new zhetapi::ml::MeanSquaredError <double> ();

	model.epochs(20, opt, imgs, exps, crit, true);
}
