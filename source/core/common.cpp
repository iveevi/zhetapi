#include <set>

#include <core/common.hpp>

namespace zhetapi {

Args args_union(const Args &a, const Args &b)
{
	std::set <std::string> xorred;
	Args un;

	for (std::string str : a)
		xorred.insert(xorred.begin(), str);
	
	for (std::string str : b)
		xorred.insert(xorred.begin(), str);

	for (std::string str : xorred)
		un.push_back(str);

	return un;
}

}
