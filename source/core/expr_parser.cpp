#include "../../engine/core/expr_parser.hpp"
#include "../../engine/core/lvalue.hpp"
#include "../../engine/core/rvalue.hpp"
#include <boost/spirit/home/support/common_terminals.hpp>

namespace zhetapi {

parser::parser() : parser::base_type(_start)
{
	// Using declarations
	using boost::phoenix::new_;
	using boost::phoenix::construct;
	using boost::spirit::qi::char_;
	using boost::spirit::qi::lit;
	using boost::spirit::qi::_val;
	using boost::spirit::qi::_1;
	using boost::spirit::qi::_2;
	using boost::spirit::qi::_3;

	_esc.add("\\a", '\a')("\\b", '\b')("\\f", '\f')("\\n", '\n')
		("\\r", '\r')("\\t", '\t')("\\v", '\v')("\\\\", '\\')
		("\\\'", '\'')("\\\"", '\"');

	// Identifier (TODO: keep outside this scope)
        _ident = char_("a-zA-Z$_") >> *char_("0-9a-zA-Z$_");

	// General string
	_str = +(_esc | (char_ - '\"'));

	// Operation parsers
	_add_operation_symbol(_plus, +);
	_add_operation_symbol(_minus, -);
	_add_operation_symbol(_times, *);
	_add_operation_symbol(_divide, /);
	_add_operation_symbol(_power, ^);
	_add_operation_symbol(_dot, @);
	_add_operation_symbol(_mod, %);
	_add_operation_symbol(_factorial, !);
	
	// Binary comparison
	_add_operation_symbol(_eq, ==);
	_add_operation_symbol(_neq, !=);
	_add_operation_symbol(_ge, >);
	_add_operation_symbol(_le, <);
	_add_operation_symbol(_geq, >=);
	_add_operation_symbol(_leq, <=);

	// Boolean ooperations
	_add_operation_symbol(_or, ||);
	_add_operation_symbol(_and, &&);

	// Unary increment/decrement
	_add_operation_heter_symbol(_post_incr, ++, p++);
	_add_operation_heter_symbol(_post_decr, --, p--);
	
	_add_operation_heter_symbol(_pre_incr, ++, r++);
	_add_operation_heter_symbol(_pre_decr, --, r--);

	// Miscellaenous
	_add_operation_symbol(_attribute, .);

	/*
	 * Represents a binary operation of lowest priotrity.
	 * These are used to combine terms into expressions.
	 * Exmaples of such operations are addition,
	 * subtraction and the dot product.
	 */
	_t0_bin = _plus | _minus | _dot | _mod
			| _eq | _neq | _geq
			| _leq | _ge | _le
			| _or | _and;

	/*
	 * Represents a binary operation of second to lowest
	 * priority. Connects factors into term. Examples are
	 * multiplication and division.
	 */
	_t1_bin = _times | _divide;
	
	/*
	 * Represents other binary operations with precedence
	 * higher than that of _t1_bin. Examples are the
	 * exponentiation operation.
	 */
	_t2_bin = _attribute | _power;

	_t_post = _post_incr | _post_decr;
	_t_pre = _pre_incr | _pre_decr;

	// Reals
	_z = boost::spirit::qi::long_long;
	_r = boost::spirit::qi::real_parser <R, boost::spirit::qi::strict_real_policies <R>> ();
	
	// Complex
	_cz = (_z >> 'i') [
		_val = construct <CmpZ> (0, _1)
	];
	
	_cr = (_r >> 'i') [
		_val = construct <CmpR> (0, _1)
	];

	// Vectors
	_vz_inter = _z % ',';
	_vr_inter = _r % ',';
	_vcz_inter = _cz % ',';
	_vcr_inter = _cr % ',';

	// TODO: disregard inters
	_vz = ('[' >> _vz_inter >> ']') [
		_val = _1
	];
	
	_vr = ('[' >> _vr_inter >> ']') [
		_val = _1
	];
	
	_vcz = ('[' >> _vcz_inter >> ']') [
		_val = _1
	];
	
	_vcr = ('[' >> _vcr_inter >> ']') [
		_val = _1
	];

	// Matrix
	_mz_inter = _vz % ',';
	_mr_inter = _vr % ',';
	_mcz_inter = _vcz % ',';
	_mcr_inter = _vcr % ',';
	
	_mz = ('[' >> _mz_inter >> ']') [
		_val = _1
	];
	
	_mr = ('[' >> _mr_inter >> ']') [
		_val = _1
	];
	
	_mcz = ('[' >> _mcz_inter >> ']') [
		_val = _1
	];
	
	_mcr = ('[' >> _mcr_inter >> ']') [
		_val = _1
	];
	
	// Token parsers

	// Reals
	_o_str = lit('\"') >> _str [
		_val = new_ <OpS> (_1)
	] >> lit('\"');

	_o_z = _z [
		_val = new_ <OpZ> (_1)
	];

	_o_r = _r [
		_val = new_ <OpR> (_1)
	];
	
	// Complex
	_o_cz = _cz [
		_val = new_ <OpCmpZ> (_1)
	];

	_o_cr = _cr [
		_val = new_ <OpCmpR> (_1)
	];

	// Vector (whats all this mess??)
	_o_vz = _vz [
		_val = new_ <OpVecZ> (_1)
	];
	
	_o_vr = _vr [
		_val = new_ <OpVecR> (_1)
	];
	
	_o_vcz = _vcz [
		_val = new_ <OpVecCmpZ> (_1)
	];
	
	_o_vcr = _vcr [
		_val = new_ <OpVecCmpR> (_1)
	];
	// Matrix
	_o_mz = _mz [
		_val = new_ <OpMatZ> (_1)
	];
	
	_o_mr = _mr [
		_val = new_ <OpMatR> (_1)
	];
	
	_o_mcz = _mcz [
		_val = new_ <OpMatCmpZ> (_1)
	];
	
	_o_mcr = _mcr [
		_val = new_ <OpMatCmpR> (_1)
	];

	// Nodes
	_node_pack = _start % ',' | boost::spirit::qi::eps;

	_collection =  ('{' >> _node_pack >> '}') [
		_val = new_ <node_list> (_1)
	];

	/*
	 * Pure numerical Operands, representing the 18
	 * primitive types of computation. The exceptions are
	 * rational numbers, which are excluded so that the
	 * order of operations can be applied. This exclusion
	 * should not cause any trouble, as integer division
	 * will yield a rational result.
	 */
	_node_opd = (
			_collection
			| _o_str
			| _o_cr | _o_cz
			| _o_r | _o_z
			| _o_vcr | _o_vcz
			| _o_vr | _o_vz
			| _o_mcr | _o_mcz
			| _o_mr | _o_mz

			// | _o_vcgr
			// | _o_vgr
			// | _o_mcgr
			// | _o_mgr
		) [
		_val = construct <node> (_1,
				std::vector <node> {})
	];

	// Rvalue and lvalues nodes
	_node_rvalue = _ident [
		_val = construct <node> (
			new_ <rvalue> (_1)
		)
	];
	
	_node_lvalue = _ident [
		_val = construct <node> (
			new_ <lvalue> (_1)
		)
	];

	/*
	 * A variable cluster, which is just a string of
	 * characters. The expansion/unpakcing of this variable
	 * cluster is done in the higher node_manager class,
	 * where access to the engine object is present.
	 */
	_node_var = (
			// Empty call
			(_ident >> '(' >> ')') [
				_val = construct <node> (
					new_ <variable_cluster> (_1),
					node(blank_token())
				)
			]

			| (_ident >> '(' >> _node_pack >> ')') [
				_val = construct <node> (
					new_ <variable_cluster> (_1),
					_2
				)
			]

			// Index
			| (_ident >> '[' >> _node_expr >> ']') [
				_val = construct <node> (
					new operation_holder("[]"),
					construct <node> (new_ <variable_cluster> (_1)),
					_2
				)
			]

			| _ident [
				_val = construct <node> (new_ <variable_cluster> (_1), std::vector <node> {})
			]
		);

	/*
	 * Represents a parenthesized expression.
	 */
	_node_prth = "(" >> _start [_val = _1] >> ")";

	/*
	 * Represents a repeatable factor. Examples are
	 * variables and parenthesized expressions. The reason
	 * is because terms like x3(5 + 3) are much more awkward
	 * compared to 3x(5 + 3)
	 */
	// TODO: the or is useless
	_node_rept = _node_var | _node_prth;
	
	/*
	 * Represents a series of parenthesized expression,
	 * which are mutlilpied through the use of
	 * juxtaposition.
	 */
	_node_prep = _node_rept [_val = _1] >> *(
		(_node_rept) [_val = construct
		<node> (new_
			<operation_holder>
			(std::string("*")), _val, _1)]
	);

	/*
	 * Represents a part of a term. For example, in the term
	 * 3x, 3 and x are both collectibles.
	 */
	_node_factor = (
		// _collection [_val = _1]

		// TODO: must rearrange attribute chains

		// TODO: clean this up for the love of god
		(_node_rept >> _power >> (_node_rept | _node_opd)) [
			_val = construct <node> (_2, _1, _3)
		]

		| (_node_opd >> _factorial) [
			_val = construct <node> (
				_2,
				_1
			)
		]

		| _node_prep [_val = _1]
	);
	
	_attr = _ident [
		_val = construct <node> (
			new_ <variable_cluster> (_1)
		)
	];

	/*
	 * Represents a term as in any mathematical expression.
	 * Should be written without addition or subtraction
	 * unless in parenthesis.
	 */
	_node_term = (
		(_node_opd >> _node_factor) [
			_val = construct <node> (
				new operation_holder("*"), _1, _2
			)
		]

		| (_node_opd >> _t1_bin >> _node_factor) [
			_val = construct <node> (_2, _1, _3)
		]
		
		| (_node_factor >> _attribute >> _node_term) [
			_val = construct <node> (_2, _1, _3)
		]
			
		| (_t_pre >> _node_var) [
			_val = construct <node> (_1, _2)
		]

		| (_node_var >> _t_post) [
			_val = construct <node> (_2, _1)
		]

		| (_minus >> _node_term) [
			_val = construct <node> (
				new operation_holder("*"), _2, node(new OpZ(-1))
			)
		]

		// TODO: 3rd term should be node_term (right recursion)
		| _node_factor [_val = _1] >> *(
			(_t1_bin >> _node_factor) [
				_val = construct <node> (_1, _val, _2)
			]
		)
		
		// TODO: manually adding powers for operands
		| (_node_opd >> _power >> _node_opd) [
			_val = construct <node> (_2, _1, _3)
		]

		// TODO: REALLY NEED TO FIX GENERALIZATION WITH OPERANDS
		| _node_opd [_val = _1]  >> *(
			(_t1_bin >> _node_term) [
				_val = construct <node> (_1, _val, _2)
			]
		)

		| _node_opd [_val = _1]
	);

	/*
	 * A full expression or function definition.
	 */
	_node_expr = (
			// TODO: use expression instead of rvalue
			(_node_lvalue >> "in" >> _node_term) [
				_val = construct <node> (
					nullptr, l_generator_in, _1, _2
				)
			]

			// TODO: add "in" for expr in expr (actually add as a
			// node_factor 
			
			| _node_term [_val = _1] >> *(
				(_t0_bin >> _node_term) [
					_val = construct <node> (_1, _val, _2)
				]
			)
		);
	
	// Entry point
	_start = _node_expr;

	// Naming rules
	_start.name("start");

	_node_expr.name("node expression");
	_node_term.name("node term");
	_node_opd.name("node Operand");
	_node_pack.name("node pack");
	_node_rept.name("Rept node");
	_node_factor.name("Factor node");

	_plus.name("addition");
	_minus.name("subtraction");
	_times.name("multiplication");
	_divide.name("division");
	_power.name("exponentiation");
	_attribute.name("attribute/method");

	_collection.name("collection");
	
	_o_str.name("literal operand");

	_o_z.name("integer operand");
	// _o_q.name("rational operand");
	_o_r.name("real operand");
	_o_cz.name("complex integer operand");
	// _o_cq.name("complex rational operand");
	_o_cr.name("complex real operand");
	
	_o_vz.name("vector integer operand");
	_o_vr.name("vector real operand");
	_o_vcz.name("vector complex integer operand");
	_o_vcr.name("vector complex real operand");
	
	_o_mz.name("matrix integer Operand");
	_o_mr.name("matrix real Operand");
	_o_mcz.name("matrix complex integer Operand");
	_o_mcr.name("matrix complex real Operand");

	_str.name("literal");

	_z.name("integer");
	_r.name("real");
	_cz.name("complex integer");
	_cr.name("complex real");
	
	_vz.name("vector integer");
	_vr.name("vector real");
	_vcz.name("vector complex integer");
	_vcr.name("vector complex real");
	
	_mz.name("matrix integer");
	_mr.name("matrix real");
	_mcz.name("matrix complex integer");
	_mcr.name("matrix complex real");
	
	_vz_inter.name("intermediate vector integer");
	_vr_inter.name("intermediate vector real");
	_vcz_inter.name("intermediate vector complex integer");
	_vcr_inter.name("intermediate vector complex real");
	
	_mz_inter.name("intermediate matrix integer");
	_mr_inter.name("intermediate matrix real");
	_mcz_inter.name("intermediate matrix complex integer");
	_mcr_inter.name("intermediate matrix complex real");

// #define ZHP_DEBUG
#ifdef ZHP_DEBUG

	debug(_start);

	debug(_node_expr);
	debug(_node_term);
	debug(_node_opd);
	debug(_node_rept);
	debug(_node_factor);
	//debug(_node_pack);

	// debug(_plus);
	// debug(_minus);
	debug(_times);
	// debug(_divide);
	debug(_power);
	// debug(_attr);
	// debug(_collection);
	
	// debug(_o_str);

	debug(_o_z);
	/* debug(_o_q);	// o_q is useless
	debug(_o_r);
	debug(_o_cz);
	debug(_o_cq);
	debug(_o_cr);

	debug(_o_vz);
	debug(_o_vq);
	debug(_o_vr);
	debug(_o_vgq);
	debug(_o_vgr);
	debug(_o_vcz);
	debug(_o_vcq);
	debug(_o_vcr);
	debug(_o_vcgq);
	debug(_o_vcgr);

	debug(_o_mz);
	debug(_o_mq);
	debug(_o_mr);
	debug(_o_mgq);
	debug(_o_mgr);
	debug(_o_mz);
	debug(_o_mcq);
	debug(_o_mcr);
	debug(_o_mcgq);
	debug(_o_mcgr);

	debug(_str); */
	
	debug(_z);
	/* debug(_q);
	debug(_r);
	debug(_gq);
	debug(_gr);
	debug(_cz);
	debug(_cq);
	debug(_cr);
	debug(_cgq);
	debug(_cgr);

	debug(_vz);
	debug(_vq);
	debug(_vr);
	debug(_vgq);
	debug(_vgr);
	debug(_vcz);
	debug(_vcq);
	debug(_vcr);
	debug(_vcgq);
	debug(_vcgr);

	debug(_mz);
	debug(_mq);
	debug(_mr);
	debug(_mgq);
	debug(_mgr);
	debug(_mz);
	debug(_mcq);
	debug(_mcr);
	debug(_mcgq);
	debug(_mcgr); */

#endif

}

}
