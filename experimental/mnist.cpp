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

int main()
{
	// MSE function (5 inputs)
	Variable x;
	Variable y;

	auto mse = square(norm(x - y))/Constant(5);
	auto true_dmse = Constant(2) * (x - y)/Constant(5);
	auto dmse = mse.differentiate(0);

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
}
