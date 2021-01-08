#include "io.hpp"

ZHETAPI_REGISTER(__zhp_std_fprint)
{
	for (Token *tptr : inputs)
		cout << tptr->str();
	
	return nullptr;
}

ZHETAPI_REGISTER(__zhp_std_fprintln)
{
	for (Token *tptr : inputs)
		cout << tptr->str();
	
	cout << "\n";
	
	return nullptr;
}