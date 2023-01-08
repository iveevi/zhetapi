#include <iostream>
#include <map>

#include <unistd.h>
#include <sys/stat.h>

#include "../include/autograd/activation.hpp"
#include "../include/autograd/autograd.hpp"
#include "../include/autograd/ml.hpp"
#include "../include/autograd/optimizer.hpp"
#include "../include/autograd/train.hpp"
#include "../include/common.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

// Files required
static const std::map <std::string, std::string> files {
	{
		"train-images-idx3-ubyte",
		"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
	},

	{
		"train-labels-idx1-ubyte",
		"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
	},

	{
		"t10k-images-idx3-ubyte",
		"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
	},

	{
		"t10k-labels-idx1-ubyte",
		"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
	}
};

// Check if file exists
bool file_exists(const std::string &path)
{
	struct stat buffer;
	return (stat(path.c_str(), &buffer) == 0);
}

#ifdef _OPENMP
#define OPENMP_ENABLED 1
#else
#define OPENMP_ENABLED 0
#endif

int main()
{
	const size_t TRAIN_IMAGES = 60000;
	const size_t VALIDATION_IMAGES = 100;
	const size_t DIMENSIONS = 784;

	std::cout << "Backend: CPU, OpenMP: " << OPENMP_ENABLED << std::endl;

	// TODO: try to use a single tensor for all data, then splice it
	const Constant::shape_type IMAGE_SHAPE = {DIMENSIONS};
	const Constant::shape_type LABEL_SHAPE = {10};

	// MSE function (5 inputs)
	Variable x;
	Variable y;

	/* auto loss = -1.0f * autograd::dot(autograd::log(x), y);
	auto dloss = (-1.0f * y/x).refactored(x, y); */

	auto loss = square(length(x - y))/Constant {10};
	auto dloss = 2 * (x - y)/Constant {10};

	std::cout << "Loss:\n" << loss.summary() << std::endl;
	std::cout << "dLoss:\n" << dloss.summary() << std::endl;

	// Model
	auto model = ml::dense(DIMENSIONS, 30)(x);
	model = ml::sigmoid(model);
	model = ml::dense(30, 10)(model);
	model = ml::softmax(model);

	std::cout << "\nModel:\n" << model.summary() << std::endl;

	// Optimizer
	auto optimizer = ml::Adam(model.parameters(), 0.01);

	// First load the MNIST dataset
	system("mkdir -p data");
	for (auto &file : files) {
		if (!file_exists("data/" + file.first)) {
			std::cout << "Downloading " << file.second << std::endl;
			system(("wget " + file.second).c_str());
			system(("gunzip " + file.first + ".gz").c_str());
			system(("mv " + file.first + " data/").c_str());
		} else {
			std::cout << "Found " << file.first << std::endl;
		}
	}

	std::cout << "\nLoading MNIST dataset..." << std::endl;

	// Load the data
	ml::Data train_data;
	ml::Data validation_data;

	std::vector <Constant> train_labels;
	std::vector <Constant> validation_labels;

	std::ifstream f_train_images("data/train-images-idx3-ubyte");
	std::ifstream f_validation_images("data/t10k-images-idx3-ubyte");

	std::ifstream f_train_labels("data/train-labels-idx1-ubyte");
	std::ifstream f_validation_labels("data/t10k-labels-idx1-ubyte");

	// Read the headers
	char header[16];

	f_train_images.read(header, 16);
	f_validation_images.read(header, 16);

	f_train_labels.read(header, 8);
	f_validation_labels.read(header, 8);

	// Read the data
	for (int i = 0; i < TRAIN_IMAGES; i++) {
		// Read the image
		unsigned char image[DIMENSIONS];
		std::vector <float> image_data;

		f_train_images.read((char *) image, DIMENSIONS);
		for (int j = 0; j < DIMENSIONS; j++)
			image_data.push_back(image[j]/255.0f);

		train_data.push_back({
			Constant {IMAGE_SHAPE, image_data}
		});

		// Read the label
		unsigned char label;
		f_train_labels.read((char *) &label, 1);

		train_labels.push_back(
			Constant {LABEL_SHAPE,
				[&](size_t i) {
					return i == label ? 1 : 0;
				}
			}
		);
	}

	for (int i = 0; i < VALIDATION_IMAGES; i++) {
		// Read the image
		unsigned char image[DIMENSIONS];
		std::vector <float> image_data;

		f_validation_images.read((char *) image, DIMENSIONS);
		for (int j = 0; j < DIMENSIONS; j++)
			image_data.push_back(image[j]/255.0f);

		validation_data.push_back({
			Constant {IMAGE_SHAPE, image_data}
		});

		// Read the label
		unsigned char label;
		f_validation_labels.read((char *) &label, 1);

		validation_labels.push_back(
			Constant {LABEL_SHAPE,
				[&](size_t i) {
					return i == label ? 1 : 0;
				}
			}
		);
	}

	// Validator
	auto validator = [](const Constant &a, const Constant &b) {
		int ai = argmax(a);
		int bi = argmax(b);
		return ai == bi;
	};

	std::cout << "Training data loaded" << std::endl;
	std::cout << "\tcurrent accuracy: " << ml::accuracy(model, train_data, train_labels, validator) << "\n" << std::endl;
	std::cout << "\toutput on input 0 = " << model(train_data[0]) << std::endl;
	std::cout << "\tlabel on input 0 = " << train_labels[0] << std::endl;
	std::cout << "\tloss = " << loss(model(train_data[0]).flat(), train_labels[0]) << std::endl;
	std::cout << "\tmatch? " << validator(model(train_data[0]), train_labels[0]) << std::endl;

	auto training_suite = ml::TrainingSuite {
		.loss = loss,
		.dloss = dloss,
		.iterations = 100,
		.batch_size = 100,
		.reporter = std::make_shared <ml::Validate> (validation_data, validation_labels, validator)
	};

	ml::fit(model, train_data, train_labels, optimizer, training_suite);

	std::cout << "\n\nTraining finished" << std::endl;
	std::cout << "\tcurrent accuracy: " << ml::accuracy(model, train_data, train_labels, validator) << std::endl;
	std::cout << "\toutput on input 0 = " << model(train_data[0]) << std::endl;
	std::cout << "\tlabel on input 0 = " << train_labels[0] << std::endl;
	std::cout << "\tmatch? " << validator(model(train_data[0]), train_labels[0]) << std::endl;

	// TODO: multithreaded training

	// TODO: some way to weight the gradients for each input (maybe by error)
	// TODO: learning rate scheduling
	// TODO: dropout and regularization
	// TODO: some method to propogate parameters through ftunctions,
	// ie. {"dropout", 0.5}, {"batch_norm", true} (a map <string, float> for now)

}
