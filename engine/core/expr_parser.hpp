#ifndef EXPR_PARSER_H_
#define EXPR_PARSER_H_

// C/C++ headers
#include <fstream>
#include <string>

// Boost headers
#include <boost/config/warning_disable.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>
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
	
	Rule <Siter, node (), Space>		_start;
	
	// Nodes
	Rule <Siter, node (), Space>		_node_expr;
	Rule <Siter, node (), Space>		_node_term;
	Rule <Siter, node (), Space>		_node_factor;
	Rule <Siter, node (), Space>		_node_prep;
	Rule <Siter, node (), Space>		_node_rept;
	Rule <Siter, node (), Space>		_node_prth;
	Rule <Siter, node (), Space>		_node_var;
	Rule <Siter, node (), Space>		_node_rvalue;
	Rule <Siter, node (), Space>		_node_lvalue;
	Rule <Siter, node (), Space>		_node_opd;

	// Helpers
	Rule <Siter, node(), Space>		_attr;

	// Parameter pack
	Rule <Siter, V1 <node> (), Space>	_node_pack;

	// Identifiers
	NSRule <Siter, std::string ()>		_ident;

	// Operations
	Rule <Siter, Token *(), Space>		_t0_bin;
	Rule <Siter, Token *(), Space>		_t1_bin;
	Rule <Siter, Token *(), Space>		_t2_bin;

	Rule <Siter, Token *(), Space>		_t_pre;
	Rule <Siter, Token *(), Space>		_t_post;

	Rule <Siter, Token *(), Space>		_eq;
	Rule <Siter, Token *(), Space>		_neq;
	Rule <Siter, Token *(), Space>		_ge;
	Rule <Siter, Token *(), Space>		_le;
	Rule <Siter, Token *(), Space>		_leq;
	Rule <Siter, Token *(), Space>		_geq;

	Rule <Siter, Token *(), Space>		_or;
	Rule <Siter, Token *(), Space>		_and;

	Rule <Siter, Token *(), Space>		_post_incr;
	Rule <Siter, Token *(), Space>		_post_decr;
	
	Rule <Siter, Token *(), Space>		_pre_incr;
	Rule <Siter, Token *(), Space>		_pre_decr;

	Rule <Siter, Token *(), Space>		_plus;
	Rule <Siter, Token *(), Space>		_minus;
	Rule <Siter, Token *(), Space>		_dot;
	Rule <Siter, Token *(), Space>		_mod;
	Rule <Siter, Token *(), Space>		_factorial;
	
	Rule <Siter, Token *(), Space>		_times;
	Rule <Siter, Token *(), Space>		_divide;
	
	Rule <Siter, Token *(), Space>		_power;
	Rule <Siter, Token *(), Space>		_attribute;
	Rule <Siter, Token *(), Space>		_collection;

	// Operands

	// Token parsers
	NSRule <Siter, Token *()>		_o_str;

	Rule <Siter, Token *(), Space>		_o_z;
	Rule <Siter, Token *(), Space>		_o_r;
	Rule <Siter, Token *(), Space>		_o_cz;
	Rule <Siter, Token *(), Space>		_o_cr;
	
	Rule <Siter, Token *(), Space>		_o_vz;
	Rule <Siter, Token *(), Space>		_o_vr;
	Rule <Siter, Token *(), Space>		_o_vcz;
	Rule <Siter, Token *(), Space>		_o_vcr;
	
	Rule <Siter, Token *(), Space>		_o_mz;
	Rule <Siter, Token *(), Space>		_o_mr;
	Rule <Siter, Token *(), Space>		_o_mcz;
	Rule <Siter, Token *(), Space>		_o_mcr;
	
	// Type parsers
	NSRule <Siter, std::string ()>		_str;

	Rule <Siter, Z (), Space>		_z;
	Rule <Siter, R (), Space>		_r;
	Rule <Siter, CmpZ (), Space>		_cz;
	Rule <Siter, CmpR (), Space>		_cr;
	
	Rule <Siter, V1 <Z> (), Space>		_vz;
	Rule <Siter, V1 <R> (), Space>		_vr;
	Rule <Siter, V1 <CmpZ> (), Space>	_vcz;
	Rule <Siter, V1 <CmpR> (), Space>	_vcr;
	
	Rule <Siter, V2 <Z> (), Space>		_mz;
	Rule <Siter, V2 <R> (), Space>		_mr;
	Rule <Siter, V2 <CmpZ> (), Space>	_mcz;
	Rule <Siter, V2 <CmpR> (), Space>	_mcr;

	// Vector and matrix intermediates (TODO: remove)
	Rule <Siter, V1 <Z> (), Space>		_vz_inter;
	Rule <Siter, V1 <R> (), Space>		_vr_inter;
	Rule <Siter, V1 <CmpZ> (), Space>	_vcz_inter;
	Rule <Siter, V1 <CmpR> (), Space>	_vcr_inter;
	
	Rule <Siter, V2 <Z> (), Space>		_mz_inter;
	Rule <Siter, V2 <R> (), Space>		_mr_inter;
	Rule <Siter, V2 <CmpZ> (), Space>	_mcz_inter;
	Rule <Siter, V2 <CmpR> (), Space>	_mcr_inter;
public:
	parser();
};

}

#endif
