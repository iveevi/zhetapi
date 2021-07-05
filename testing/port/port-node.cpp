#include "port.hpp"

// TODO: add another test for checking hanlding of invalid input
TEST(compile_const_exprs)
{
	using namespace zhetapi;

	// Context
	Engine *ctx = new Engine(true);

	// List of compiled expressions
	vector <std::string> exprs {
		// Operands
		"564", "7/5", "false", "\"my string\"",
		"[1, 2, 3, 5, 8]", "[[1, 2, 3], [4, 5, 6]]",
		// First come the binary operations
		"7 + 5", "7 - 5", "7 * 5", "7 / 5"
	};

	// List of their values
	vector <Token *> values {
		// Operands
		new OpZ(564), new OpQ(Q(7, 5)), new OpB(false),
		new OpS("my string"), new OpVecZ(VecZ({1, 2, 3, 5, 8})),
		new OpMatZ(MatZ({{1, 2, 3}, {4, 5, 6}})),
		// Binary operations
		new OpZ(12), new OpZ(2), new OpZ(35), new OpQ(Q(7, 5))
	};

	// Comparing the values
	size_t count = 0;

	size_t size = exprs.size();
	for (size_t i = 0; i < size; i++) {
		node_manager nm;
		Token *tptr;
	
		// Construction
		try {
			nm = node_manager(ctx, exprs[i]);
		} catch (const node_manager::bad_input &e) {
			oss << err << "\tBad input exception was thrown with \""
				<< exprs[i] << "\"" << endl;
			continue;
		} catch (const engine_base::unknown_op_overload &e) {
			oss << err << "\tUnknown operation overload with \""
				<< exprs[i] << "\"" << endl;
			oss << e.what() << endl;
			oss << "Tree:" << endl;
			nm.print();
			continue;
		}

		// Evaluation
		try {
			tptr = nm.value(ctx);
		} catch (const engine_base::unknown_op_overload &e) {
			oss << err << e.what() << endl;
			oss << "Tree:" << endl;
			nm.print();
			continue;
		}

		oss << "Value of \"" << exprs[i] << "\" = "
			<< (tptr ? tptr->dbg_str() : "[Null]")
			<< " <=> " << values[i]->dbg_str() << endl;
		
		if (tokcmp(tptr, values[i])) {
			oss << ok << "\tValues are equal." << endl;
			count++;
		} else {
			oss << err << "\tValues are mismatching." << endl;
		}
	}

	return (count == size);
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

	return (count == size);
}
