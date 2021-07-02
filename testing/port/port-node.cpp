#include "port.hpp"

// Kind of a stupid test given the one right after,
// but oh well, better have more tests than none :P
TEST(compile_operand)
{
	using namespace zhetapi;

	// Total number of tests here
	int count = 5;

	// Context
	Engine *ctx = new Engine(true);

	// Manual trees
	node n1(new OpZ(564));
	node n2(new OpR(1.618));
	node n3(new OpB(false));
	node n4(new OpS("my string"));
	/* TODO: fix this
	node n5(
		new Operand <({
			new OpZ(1), new OpR(3.14), new OpB(true)
		})
	); */

	// Compiled trees
	node_manager nm1(ctx, "564");
	node_manager nm2(ctx, "1.618");
	node_manager nm3(ctx, "false");
	node_manager nm4(ctx, "\"my string\"");
	// node_manager nm5(ctx, "{1, 3.14, true}");

	// Integer
	if (node::loose_match(n1, nm1.tree())) {
		oss << ok << "Integer trees" << reset << endl;
		count--;
	} else {
		oss << err << "Integer trees" << reset << endl;
		oss << "Manual:" << endl;
		n1.print();
		oss << "Compiled:" << endl;
		nm1.print();
	}
	
	// Real
	if (node::loose_match(n2, nm2.tree())) {
		oss << ok << "Real trees" << reset << endl;
		count--;
	} else {
		oss << err << "Real trees" << reset << endl;
		oss << "Manual:" << endl;
		n2.print();
		oss << "Compiled:" << endl;
		nm2.print();
	}
	
	// Boolean
	if (node::loose_match(n3, nm3.tree())) {
		oss << ok << "Boolean trees" << reset << endl;
		count--;
	} else {
		oss << err << "Boolean trees" << reset << endl;
		oss << "Manual:" << endl;
		n3.print();
		oss << "Compiled:" << endl;
		nm3.print();
	}
	
	// String
	if (node::loose_match(n4, nm4.tree())) {
		oss << ok << "String trees" << reset << endl;
		count--;
	} else {
		oss << err << "String trees" << reset << endl;
		oss << "Manual:" << endl;
		n4.print();
		oss << "Compiled:" << endl;
		nm4.print();
	}
	
	/* Collection
	if (node::loose_match(n5, nm5.tree())) {
		oss << ok << "Collection trees" << reset << endl;
		count--;
	} else {
		oss << err << "Collection trees" << reset << endl;
		oss << "Manual:" << endl;
		n5.print();
		oss << "Compiled:" << endl;
		nm5.print();
	} */

	return !count;
}

TEST(compile_const_exprs)
{
	using namespace zhetapi;

	// Context
	Engine *ctx = new Engine();

	// List of compiled expressions
	vector <std::string> exprs {
		// First come the binary operations
		"7 + 5", "7 - 5", "7 * 5", "7 / 5"
	};

	// List of their values
	vector <Token *> values {
		// Binary operations
		new OpZ(12), new OpZ(2), new OpZ(35), new OpQ(Q(7, 5))
	};

	// Comparing the values
	size_t count = 0;

	size_t size = exprs.size();
	for (size_t i = 0; i < size; i++) {
		node_manager nm(ctx, exprs[i]);

		Token *tptr = nm.value(ctx);
		oss << "Value of \"" << exprs[i] << "\" = "
			<< (tptr ? tptr->dbg_str() : "[Null]")
			<< " <=> " << values[i]->dbg_str() << endl;
		
		if (tokcmp(tptr, values[i]))
			oss << ok << "\tValues are equal." << endl;
		else
			oss << err << "\tValues are mismatching." << endl;
	}

	return count == size;
}

TEST(compile_var_exprs)
{
	using namespace zhetapi;

	// Context
	Engine *ctx = new Engine();

	// Common variable
	Token *vptr1 = new node_reference(nullptr, "x", 0, true);

	// List of compiled expressions
	vector <std::string> exprs {
		"x^2", "2x"
	};

	// List of their values
	vector <node> trees {
		node(new operation_holder("^"), {
			node(vptr1->copy()),
			node(new OpZ(2))
		}),
		node(new operation_holder("*"), {
			node(vptr1->copy()),
			node(new OpZ(2))
		})
	};

	// Comparing the values
	size_t count = 0;

	size_t size = exprs.size();
	for (size_t i = 0; i < size; i++) {
		node_manager nm(ctx, exprs[i], {"x"});
		
		if (node::loose_match(nm.tree(), trees[i])) {
			oss << ok << "\tTrees are matching." << endl;
			count++;
		} else {
			oss << err << "\tTrees are mismatching." << endl;
			oss << "Precompiled (manual):" << endl;
			trees[i].print();
			oss << "Compiled:" << endl;
			nm.print();
		}
	}

	return count == size;
}