#include "../../engine/core/enode_parser.hpp"

namespace zhetapi {

// Static
static EnodeParser eparser;

// Macros to simplify
#define op_lit(var, str, code)				\
	var = boost::spirit::qi::lit(#str) [		\
		boost::spirit::qi::_val = code		\
	]

EnodeParser::EnodeParser()
		: EnodeParser::base_type(_start)
{
	using namespace boost::spirit;
	using namespace boost::phoenix;

	/* _esc.add("\\a", '\a')("\\b", '\b')("\\f", '\f')("\\n", '\n')
		("\\r", '\r')("\\t", '\t')("\\v", '\v')("\\\\", '\\')
		("\\\'", '\'')("\\\"", '\"'); */

	// Operations
	op_lit(_addition, +, l_add);
	op_lit(_subtraction, -, l_sub);
	op_lit(_multiplication, *, l_mul);
	op_lit(_division, /, l_div);

	// Defining precedences
	_t0_bin = _addition | _subtraction;
	_t1_bin = _multiplication | _division;

	// Primitives
	_int = qi::long_long [
		_val = construct <Primitive> (_1)
	];
	
	_double = (qi::real_parser <long double, qi::strict_real_policies <long double>> ()) [
		_val = construct <Primitive> (_1)
	];

	// Structural
	_primop = (_double | _int);

	// TODO: change operand to factor
	_term = (
		_primop [_val = _1] >> *(_t1_bin >> _primop) [
			_val = construct <Enode> (_1, _val, _2)
		]
	);

	_start = (
		_term [_val = _1] >> *(_t0_bin >> _term) [
			_val = construct <Enode> (_1, _val, _2)
		]
	);

// #define ENODE_PARSER_DEBUG
#ifdef ENODE_PARSER_DEBUG
	
	// Naming
	_start.name("Start");
	_term.name("Term");
	_operand.name("Operand");

	_int.name("Int");
	_double.name("Double");

	_t0_bin.name("#0 binary");
	_t1_bin.name("#1 binary");

	_addition.name("Addition");
	_subtraction.name("Subtraction");
	_multiplication.name("Multiplication");
	_division.name("Division");

	// Debugging
	debug(_start);
	debug(_term);
	debug(_operand);
	
	debug(_int);
	debug(_double);

	debug(_t0_bin);
	debug(_t1_bin);

	debug(_addition);
	debug(_subtraction);
	debug(_multiplication);
	debug(_division);

#endif

}

Enode strmake(const std::string &str)
{
	auto start = str.begin();
	auto end = str.end();

	Enode tree;

	bool good = boost::spirit::qi::phrase_parse(
			start,
			end,
			eparser,
			boost::spirit::qi::space,
			tree);

	if (!good)
		throw EnodeParser::bad_parse(str);

	return tree;
}

}
