#include "port.hpp"

bool tensor_construction_and_memory()
{
	using namespace zhetapi;

	Tensor <double> tmp;

	cout << "Default: " << tmp << endl;

	return true;
}