#include "port.hpp"

TEST(module_construction)
{
	using namespace zhetapi;

	Module m1("m1");

	Module m2("m2", {
		{"e", new OpR(exp(1))}
	});

	m2.list_attributes(oss);

	return true;
}