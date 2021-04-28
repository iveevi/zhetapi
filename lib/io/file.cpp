#include "io.hpp"

ZHETAPI_REGISTER(__zhp_std_fprint)
{
	Operand <string> *file;

	zhetapi_cast(inputs, file);

	ofstream fout(file->get());

	size_t n = inputs.size();
	for (size_t i = 1; i < n; i++)
		fout << inputs[i]->dbg_str();
	
	return nullptr;
}

// TODO: change to str() later
ZHETAPI_REGISTER(__zhp_std_fprintln)
{
	Operand <string> *file;

	zhetapi_cast(inputs, file);

	ofstream fout(file->get());

	size_t n = inputs.size();
	for (size_t i = 1; i < n; i++)
		fout << inputs[i]->dbg_str();
	
	fout << "\n";
	
	return nullptr;
}