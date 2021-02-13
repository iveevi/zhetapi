#include "port.hpp"

bool tensor_construction_and_memory(ostringstream &oss)
{
	using namespace zhetapi;

	Tensor <double> t1;

	oss << "Default: " << t1 << endl;

	Tensor <double> t2({4, 5, 6});

	oss << "Dimension constructor: " << t2 << endl;

	return true;
}
