#include "port.hpp"

zhetapi::StringFeeder src1(R"(
x = 12
y = 2, z = 24
)");

zhetapi::StringFeeder src2(R"(
x = 12
y = 2
z = 24
w = 786
const = 14

if (true)
	x = 24

if (true) {
	w = 907

	y = 37, z = 76
}

if (false)
	const = 25

px = 0
py = 2

if (px^2 + py^2 == 4)
	px = 24

if (x^2 + 5^2 == 12^2)
	x = 5

if (x^2 + 10^2 == 26^2) {
	py = 27

	px = py + px
}
)");

TEST(parsing_global_assignment)
{
	using namespace zhetapi;

	Engine *context = new Engine(true);

	parse_global(&src1, context);

	Token *tptr1 = context->get("x");
	if (OpZ (12) == tptr1) {
		oss << ok << " Correct value found for x." << endl;
	} else {
		oss << err << " Wrong value for x." << endl;

		return false;
	}

	Token *tptr2 = context->get("y");
	if (OpZ (2) == tptr2) {
		oss << ok << " Correct value found for y." << endl;
	} else {
		oss << err << " Wrong value for y." << endl;

		return false;
	}

	Token *tptr3 = context->get("z");
	if (OpZ (24) == tptr3) {
		oss << ok << " Correct value found for z." << endl;
	} else {
		oss << err << " Wrong value for z." << endl;

		return false;
	}

	Token *tptr4 = context->get("w");
	if (tptr4) {
		oss << err << "There was no symbol \"w\"." << endl;
		
		return false;
	} else {
		oss << ok << "The symbol \"w\" was never defined." << endl;
	}

	delete context;
	delete tptr1;
	delete tptr2;
	delete tptr3;

	return true;
}

TEST(parsing_global_branching)
{
	using namespace zhetapi;

	Engine *context = new Engine(true);

	parse_global(&src2, context);

	Token *tptr1 = context->get("x");
	if (OpZ (24) == tptr1) {
		oss << ok << " Correct value found for x." << endl;
	} else {
		oss << err << " Wrong value for x." << endl;

		return false;
	}

	Token *tptr2 = context->get("y");
	if (OpZ (37) == tptr2) {
		oss << ok << " Correct value found for y." << endl;
	} else {
		oss << err << " Wrong value for y." << endl;

		return false;
	}

	Token *tptr3 = context->get("z");
	if (OpZ (76) == tptr3) {
		oss << ok << " Correct value found for z." << endl;
	} else {
		oss << err << " Wrong value for z." << endl;

		return false;
	}

	Token *tptr4 = context->get("w");
	if (OpZ (907) == tptr4) {
		oss << ok << " Correct value found for w." << endl;
	} else {
		oss << err << " Wrong value for w." << endl;

		return false;
	}

	Token *tptr5 = context->get("const");
	if (OpZ (14) == tptr5) {
		oss << ok << " Correct value found for const." << endl;
	} else {
		oss << err << " Wrong value for const." << endl;

		return false;
	}

	Token *tptr6 = context->get("px");
	if (OpZ (51) == tptr6) {
		oss << ok << " Correct value found for px." << endl;
	} else {
		oss << err << " Wrong value for px." << endl;

		return false;
	}

	Token *tptr7 = context->get("py");
	if (OpZ (27) == tptr7) {
		oss << ok << " Correct value found for py." << endl;
	} else {
		oss << err << " Wrong value for py." << endl;

		return false;
	}

	delete context;
	delete tptr1;
	delete tptr2;
	delete tptr3;
	delete tptr4;
	delete tptr5;
	delete tptr6;
	delete tptr7;

	return true;
}