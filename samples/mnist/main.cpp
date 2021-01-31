// C/C++ headers
#include <iostream>
#include <fstream>
#include <vector>

// Engine standard headers
#include <std/activations.hpp>
#include <std/optimizers.hpp>
#include <std/erfs.hpp>

#define ZHP_ENGINE_PATH "../../engine"

// Engine headers
#include <network.hpp>
#include <training.hpp>

#define TRAIN_IMAGES	60000
#define VALID_IMAGES	10000
#define SIZE		28

using namespace std;
using namespace zhetapi;

// Global variables
ifstream train_images("train-images-idx3-ubyte", ios::binary);
ifstream train_labels("train-labels-idx1-ubyte", ios::binary);

ifstream valid_images("train-images-idx3-ubyte", ios::binary);
ifstream valid_labels("train-labels-idx1-ubyte", ios::binary);

DataSet <double> train_imgs;
DataSet <double> train_exps;

DataSet <double> valid_imgs;
DataSet <double> valid_exps;

// Pass critique
bool match(const Vector <double> &actual, const Vector <double> &expected)
{
	int mi = 0;
	for (int i = 1; i < 10; i++) {
		if (actual[mi] < actual[i])
			mi = i;
	}

	return (expected[mi] == 1);
};

// Reading images
vector <double> read_image(ifstream &fin)
{
	vector <double> pixels;

	int size = SIZE * SIZE;
	for (int k = 0; k < size; k++) {
		unsigned char temp = 0;

		fin.read((char*) &temp, sizeof(temp));

		pixels.push_back(temp);
	}
	
	return pixels;
}

// Main function
int main()
{
	// Enable loading
	ml::ZhetapiRegisterStandardActivations <double> ();

	// Load the model structure
	// model.load_json("model.json");

	// Seed generator
	srand(clock());

	// Create the model
	ml::NeuralNetwork <double> model(784, {
		ml::Layer <double> (30, new ml::Sigmoid <double> ()),
		ml::Layer <double> (10, new ml::Softmax <double> ())
	});

	// model.print();

	// Temporary variable
	unsigned int tmp;

	// First 16 bytes
	train_images.read((char *) &tmp, sizeof(tmp));
	train_images.read((char *) &tmp, sizeof(tmp));
	train_images.read((char *) &tmp, sizeof(tmp));
	train_images.read((char *) &tmp, sizeof(tmp));
	
	valid_images.read((char *) &tmp, sizeof(tmp));
	valid_images.read((char *) &tmp, sizeof(tmp));
	valid_images.read((char *) &tmp, sizeof(tmp));
	valid_images.read((char *) &tmp, sizeof(tmp));
	
	// First 8 bytes
	train_labels.read((char *) &tmp, sizeof(tmp));
	train_labels.read((char *) &tmp, sizeof(tmp));
	
	valid_labels.read((char *) &tmp, sizeof(tmp));
	valid_labels.read((char *) &tmp, sizeof(tmp));

	// Extract training data
	for (size_t i = 0; i < TRAIN_IMAGES; i++) {
		Vector <double> in = read_image(train_images);

		unsigned char actual;

		train_labels.read((char *) &actual, sizeof(actual));

		Vector <double> exp(10,
			[&](size_t i) {
				return (i == actual) ? 1.0 : 0.0;
			}
		);

		train_imgs.push_back(in);
		train_exps.push_back(exp);
	}
	
	// Extract validation data
	for (size_t i = 0; i < VALID_IMAGES; i++) {
		Vector <double> in = read_image(valid_images);

		unsigned char actual;

		valid_labels.read((char *) &actual, sizeof(actual));

		Vector <double> exp(10,
			[&](size_t i) {
				return (i == actual) ? 1.0 : 0.0;
			}
		);

		valid_imgs.push_back(in);
		valid_exps.push_back(exp);
	}

	ml::Erf <double> *cost = new ml::MeanSquaredError <double> ();
	ml::Optimizer <double> *opt = new ml::Nesterov <double> (1, 0.7);

	model.set_cost(cost);
	model.set_optimizer(opt);

	for (int i = 0; i < 100; i++)
		train_dataset_perf(model, train_imgs, train_exps, 128, cost, match, Display::batch);

	// Free resources
	delete cost;
	delete opt;
}
