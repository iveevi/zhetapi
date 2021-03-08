#include "port.hpp"

bool matrix_construction_and_memory(ostringstream &oss)
{
	using namespace zhetapi;

	Matrix <double> tmp;

	oss << "Default constructor: " << tmp << endl;

	return true;
}
