#ifndef EXPR_PARSER_H_
#define EXPR_PARSER_H_

// C/C++ headers
#include <fstream>
#include <string>

// Boost headers
#include <boost/config/warning_disable.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>
#include <boost/phoenix/function/adapt_function.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/phoenix_container.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_statement.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/qi.hpp>

// Engine headers
#include "../operand.hpp"

#include "node.hpp"
#include "node_list.hpp"
#include "operation_holder.hpp"
#include "variable_cluster.hpp"
#include "special_tokens.hpp"

namespace zhetapi {

// Parser
struct parser : boost::spirit::qi::grammar <
		std::string::const_iterator,
		node (),
		boost::spirit::qi::space_type> {
public:
	using Siter = std::string::const_iterator;

	// Qi rule (no skip)
	template <class A, class B>
	using NSRule = boost::spirit::qi::rule <A, B>;

	// Qi rule
	template <class A, class B, class C>
	using Rule = boost::spirit::qi::rule <A, B, C>;

	// Space type
	using Space = boost::spirit::qi::space_type;

	// Vector aliases
	template <class T>
	using V1 = std::vector <T>;

	template <class T>
	using V2 = std::vector <std::vector <T>>;
private:
	boost::spirit::qi::symbols
		<char const, char const>	_esc;

	// Node grammars (structural, from bottom to top)
	Rule <Siter, node (), Space>		_operand;
	Rule <Siter, node (), Space>		_closed_factor;
	Rule <Siter, node (), Space>		_full_factor;
	Rule <Siter, node (), Space>		_factor;
	Rule <Siter, node (), Space>		_term;
	Rule <Siter, node (), Space>		_simple_expression;
	Rule <Siter, node (), Space>		_start;

	// Type parsers
	Rule <Siter, Z (), Space>		_integer;
	Rule <Siter, R (), Space>		_pure_real;
	Rule <Siter, R (), Space>		_real;
	Rule <Siter, VecZ (), Space>		_vector_integer;
	Rule <Siter, VecR (), Space>		_vector_real;
	Rule <Siter, MatZ (), Space>		_matrix_integer;
	Rule <Siter, MatR (), Space>		_matrix_real;
	Rule <Siter, V1 <node> (), Space>	_vector_expr;
	Rule <Siter, node (), Space>		_partial_matrix_expr;
	Rule <Siter, V1 <node> (), Space>	_matrix_expr;
	Rule <Siter, Token *(), Space>		_collection;
	NSRule <Siter, std::string ()>		_string;
	NSRule <Siter, std::string ()>		_identifier;

	// Categories of operations
	Rule <Siter, std::string(), Space>	_term_operation;
	Rule <Siter, std::string(), Space>	_start_operation;
	Rule <Siter, std::string(), Space>	_expression_operation;
	Rule <Siter, std::string(), Space>	_post_operations;
	Rule <Siter, std::string(), Space>	_pre_operations;

	// Operations
	Rule <Siter, std::string(), Space>	_and;
	Rule <Siter, std::string(), Space>	_attribute;
	Rule <Siter, std::string(), Space>	_divide;
	Rule <Siter, std::string(), Space>	_dot;
	Rule <Siter, std::string(), Space>	_eq;
	Rule <Siter, std::string(), Space>	_exponent;
	Rule <Siter, std::string(), Space>	_factorial;
	Rule <Siter, std::string(), Space>	_ge;
	Rule <Siter, std::string(), Space>	_geq;
	Rule <Siter, std::string(), Space>	_le;
	Rule <Siter, std::string(), Space>	_leq;
	Rule <Siter, std::string(), Space>	_minus;
	Rule <Siter, std::string(), Space>	_mod;
	Rule <Siter, std::string(), Space>	_neq;
	Rule <Siter, std::string(), Space>	_or;
	Rule <Siter, std::string(), Space>	_plus;
	Rule <Siter, std::string(), Space>	_transpose;
	Rule <Siter, std::string(), Space>	_post_decr;
	Rule <Siter, std::string(), Space>	_post_incr;
	Rule <Siter, std::string(), Space>	_pre_decr;
	Rule <Siter, std::string(), Space>	_pre_incr;
	Rule <Siter, std::string(), Space>	_times;
public:
	parser(Engine *ctx);
};

}

#endif
