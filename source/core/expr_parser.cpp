#include "../../engine/core/node_manager.hpp"
#include "../../engine/core/expr_parser.hpp"
#include "../../engine/core/lvalue.hpp"
#include "../../engine/core/rvalue.hpp"
#include "../../engine/core/collection.hpp"

namespace zhetapi {

// Macros
#define _add_operation_symbol(name, str)				\
	name = boost::spirit::qi::lit(#str) [				\
		boost::spirit::qi::_val = boost::phoenix::new_		\
			<operation_holder> (std::string(#str))		\
	];

#define _add_operation_heter_symbol(name, str, act)			\
	name = boost::spirit::qi::lit(#str) [				\
		boost::spirit::qi::_val = boost::phoenix::new_		\
			<operation_holder> (std::string(#act))		\
	];

#define _new_oph(str)							\
	new zhetapi::operation_holder(str)

// TODO: will need to create two parsers
// - one for immediate parsing and evaluation (all symbols are assumed to be
// defined)
// - another for delayed execution: creating funtions/algorithms
// welp, this is for the new parser set...
parser::parser(Engine *ctx)
		: parser::base_type(_start)
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

	// Escape characters
	_esc.add("\\a", '\a')
		("\\b", '\b')
		("\\f", '\f')
		("\\n", '\n')
		("\\r", '\r')
		("\\t", '\t')
		("\\v", '\v')
		("\\\\", '\\')
		("\\\'", '\'')
		("\\\"", '\"');

	// Operation parsers
	_add_operation_symbol(_and, &&);
	_add_operation_symbol(_attribute, .);
	_add_operation_symbol(_divide, /);
	_add_operation_symbol(_dot, @);
	_add_operation_symbol(_eq, ==);
	_add_operation_symbol(_exponent, ^);
	_add_operation_symbol(_factorial, !);
	_add_operation_symbol(_ge, >);
	_add_operation_symbol(_geq, >=);
	_add_operation_symbol(_le, <);
	_add_operation_symbol(_leq, <=);
	_add_operation_symbol(_minus, -);
	_add_operation_symbol(_mod, %);
	_add_operation_symbol(_neq, !=);
	_add_operation_symbol(_or, ||);
	_add_operation_symbol(_plus, +);
	_add_operation_symbol(_times, *);
	_add_operation_symbol(_transpose, ^T);
	_add_operation_heter_symbol(_post_decr, --, p--);
	_add_operation_heter_symbol(_post_incr, ++, p++);
	_add_operation_heter_symbol(_pre_decr, --, r--);
	_add_operation_heter_symbol(_pre_incr, ++, r++);

	// Categorizing operations
	_term_operation = _times | _divide | _mod | _dot | _and;
	_start_operation = _plus | _minus | _or;
	_expression_operation = _le | _leq | _ge | _geq | _eq | _neq;
	_post_operations = _factorial | _post_decr | _post_incr;
	_pre_operations = _pre_decr | _pre_incr;

	// Type parsers (real can be integer as well)
	_integer = boost::spirit::qi::ulong_long;

	_pure_real = boost::spirit::qi::real_parser
		<R, boost::spirit::qi::strict_ureal_policies <R>> ();

	_real = _pure_real | _integer;

	// TODO: add general vectors (made of expressions)
	_vector_integer = (_integer % ',') [
		_val = construct <VecZ> (_1)
	];

	// Second priority over vector integer
	_vector_real = (_real % ',') [
		_val = construct <VecR> (_1)
	];

	// Use V2 constructor instead of vector constructor
	_matrix_integer = (('[' >> (_integer % ',') >> ']')
			[_val = _1] % ',') [
		_val = construct <MatZ> (_1)
	];

	// Use V2 constructor instead of vector constructor
	_matrix_real = (('[' >> (_real % ',') >> ']')
			[_val = _1] % ',') [
		_val = construct <MatR> (_1)
	];

	// Generic vectors and matrices
	_vector_expr = _start % ',';

	_partial_matrix_expr = ('[' >> (_start % ',') >> ']') [
		_val = construct <node> (nullptr, l_partial_matrix_expr, _1)
	];

	_matrix_expr = _partial_matrix_expr % ',';

	// Quoted strings
	_string = +(_esc | (char_ - '\"'));

	// Identifier (TODO: keep outside this scope later)
	_identifier = char_("a-zA-Z$_") >> *char_("0-9a-zA-Z$_");

	_collection = (
		lit("{}") [
			_val = new_ <Collection> (V1 <Token *> {})
		]

		| ('{' >> (_start % ',') >> '}') [
			_val = new_ <node_list> (_1)
		]
	);

	// Operands (TODO: can also be a operation of operands)
	_operand = (
		_pure_real [
			_val = construct <node> (new_ <OpR> (_1))
		]

		// TODO: does real go after?
		| _integer [
			_val = construct <node> (new_ <OpZ> (_1))
		]

		| _collection [
			_val = construct <node> (_1)
		]

		| ('\"' >> _string >> '\"') [
			_val = construct <node> (new_ <OpS> (_1))
		]

		| ('[' >> _vector_integer >> ']') [
			_val = construct <node> (new_ <OpVecZ> (_1))
		]

		| ('[' >> _vector_real >> ']') [
			_val = construct <node> (new_ <OpVecR> (_1))
		]

		| ('[' >> _matrix_integer >> ']') [
			_val = construct <node> (new_ <OpMatZ> (_1))
		]

		| ('[' >> _matrix_real >> ']') [
			_val = construct <node> (new_ <OpMatR> (_1))
		]

		// Matrix of expressions
		| ('[' >> _matrix_expr >> ']') [
			_val = construct <node> (nullptr, l_matrix_expr, _1)
		]

		// Vector of expressions
		| ('[' >> _vector_expr >> ']') [
			_val = construct <node> (nullptr, l_vector_expr, _1)
		]
	);

	// Closed factors: tighter than factors and full factors
	_closed_factor = (
		_operand [_val = _1]

		// Prioritize function with aruments (or blank)
		// TODO: extended _identifier with closed factor
		| (_identifier >> "()") [
			_val = construct <node> (
				new_ <variable_cluster> (_1),
				V1 <node> {node(blank_token())}
			)
		]

		| (_identifier >> '(' >> (_start % ',') >> ')') [
			_val = construct <node> (new_ <variable_cluster> (_1), _2)
		]

		// TODO: expand right (if immediate) here using ctx and args
		| _identifier [
			_val = construct <node> (new_ <variable_cluster> (_1))
		]

		// Parenthesized expressions
		| ('(' >> _start >> ')') [
			_val = _1
		]
	);

	// Full factors
	_full_factor = (
		(_closed_factor >> "in" >> _full_factor) [
			_val = construct <node> (nullptr, l_generator_in, _1, _2)
		]

		| (_closed_factor >> _transpose) [
			_val = construct <node> (_2, _1)
		]

		| (_closed_factor >> _exponent >> _full_factor) [
			_val = construct <node> (_2, _1, _3)
		]

		// Relax the grammar here, allows for garbage like foo.23
		| (_closed_factor >> _attribute >> _closed_factor) [
			_val = construct <node> (_2, _1, _3)
		]

		| (_closed_factor >> '[' >> _start >> ']') [
			_val = construct <node> (_new_oph("[]"), _1, _2)
		]

		| (_closed_factor >> _post_operations) [
			_val = construct <node> (_2, _1)
		]

		| (_pre_operations >> _closed_factor) [
			_val = construct <node> (_1, _2)
		]

		| _closed_factor [_val = _1]
	);

	// Factors: juxtaposition of full factors
	_factor = (
		(_full_factor >> _factor) [
			_val = construct <node> (_new_oph("*"), _1, _2)
		]

		| _full_factor [_val = _1]
	);

	// Terms: series of factors under multiplication or division
	_term = (
		(_plus >> _term) [
			_val = _2
		]

		// TODO: add a constant for -1
		| (_minus >> _term) [
			_val = construct <node> (
				_new_oph("*"), _2, node(new OpZ(-1))
			)
		]

		| (_factor >> _term_operation >> _term) [
			_val = construct <node> (_2, _1, _3)
		]

		| _factor [_val = _1]
	);

	// Simple expression
	_simple_expression = (
		(_term >> _start_operation >> _start) [
			_val = construct <node> (_2, _1, _3)
		]

		| _term [_val = _1]
	);

	// Starting expression
	_start = (
		(_simple_expression >> _expression_operation >> _start) [
			_val = construct <node> (_2, _1, _3)
		]

		| _simple_expression [_val = _1]
	);

// #define ZHETAPI_PARSER_DEBUG
#ifdef ZHETAPI_PARSER_DEBUG

	// Debugging name information
	_integer.name("Integer parser");
	_pure_real.name("Pure real (decimal) parser");
	_real.name("Real parser");
	_collection.name("Collection parser");
	_identifier.name("Indentifier parser");

	_operand.name("Operand");
	_closed_factor.name("Closed Factor");
	_full_factor.name("Full Factor");
	_factor.name("Factor");
	_term.name("Term");
	_start.name("Start");

	_plus.name("Plus");
	_times.name("Times");

	// Confirming debug
	debug(_integer);
	debug(_pure_real);
	// debug(_real);
	// debug(_collection);
	// debug(_identifier);

	// debug(_operand);
	// debug(_closed_factor);
	// debug(_full_factor);
	// debug(_factor);
	// debug(_term);
	// debug(_start);

	// debug(_plus);
	// debug(_times);

#endif
}

}
