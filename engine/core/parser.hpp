#ifndef PARSER_H_
#define PARSER_H_

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
#include <operand.hpp>

#include <core/node.hpp>
#include <core/node_list.hpp>
#include <core/operation_holder.hpp>
#include <core/variable_cluster.hpp>

#define _add_operation_symbol(name, str)				\
	name = lit(#str) [						\
		_val = phoenix::new_					\
			<operation_holder> (std::string(#str))		\
	];

#define _add_operation_heter_symbol(name, str, act)			\
	name = lit(#str) [						\
		_val = phoenix::new_					\
			<operation_holder> (std::string(#act))		\
	];

namespace zhetapi {

typedef std::string::const_iterator siter;

using namespace boost;

using namespace boost::spirit;
using namespace boost::phoenix;

// Parser
struct parser : qi::grammar <siter, node (), qi::space_type> {

	parser();

	qi::symbols <char const, char const>				_esc;
	
	qi::rule <siter, node (), qi::space_type>			_start;
	
	// Nodes
	qi::rule <siter, node (), qi::space_type>			_node_expr;
	qi::rule <siter, node (), qi::space_type>			_node_term;
	qi::rule <siter, node (), qi::space_type>			_node_factor;
	qi::rule <siter, node (), qi::space_type>			_node_prep;
	qi::rule <siter, node (), qi::space_type>			_node_rept;
	qi::rule <siter, node (), qi::space_type>			_node_prth;
	qi::rule <siter, node (), qi::space_type>			_node_var;
	qi::rule <siter, node (), qi::space_type>			_node_opd;

	// Helpers
	qi::rule <siter, node(), qi::space_type>			_attr;

	// Parameter pack
	qi::rule <siter, std::vector <node> (), qi::space_type>	_node_pack;

	// Identifiers
	qi::rule <siter, std::string ()>				_ident;

	// Operations
	qi::rule <siter, Token *(), qi::space_type>			_t0_bin;
	qi::rule <siter, Token *(), qi::space_type>			_t1_bin;
	qi::rule <siter, Token *(), qi::space_type>			_t2_bin;

	qi::rule <siter, Token *(), qi::space_type>			_t_pre;
	qi::rule <siter, Token *(), qi::space_type>			_t_post;

	qi::rule <siter, Token *(), qi::space_type>			_eq;
	qi::rule <siter, Token *(), qi::space_type>			_neq;
	qi::rule <siter, Token *(), qi::space_type>			_ge;
	qi::rule <siter, Token *(), qi::space_type>			_le;
	qi::rule <siter, Token *(), qi::space_type>			_leq;
	qi::rule <siter, Token *(), qi::space_type>			_geq;

	qi::rule <siter, Token *(), qi::space_type>			_post_incr;
	qi::rule <siter, Token *(), qi::space_type>			_post_decr;
	
	qi::rule <siter, Token *(), qi::space_type>			_pre_incr;
	qi::rule <siter, Token *(), qi::space_type>			_pre_decr;

	qi::rule <siter, Token *(), qi::space_type>			_plus;
	qi::rule <siter, Token *(), qi::space_type>			_minus;
	qi::rule <siter, Token *(), qi::space_type>			_dot;
	qi::rule <siter, Token *(), qi::space_type>			_mod;
	
	qi::rule <siter, Token *(), qi::space_type>			_times;
	qi::rule <siter, Token *(), qi::space_type>			_divide;
	
	qi::rule <siter, Token *(), qi::space_type>			_power;

	qi::rule <siter, Token *(), qi::space_type>			_attribute;

	qi::rule <siter, Token *(), qi::space_type>			_collection;

	// Operands

	// Token parsers
	qi::rule <siter, Token *()>					_o_str;

	qi::rule <siter, Token *(), qi::space_type>			_o_z;
	qi::rule <siter, Token *(), qi::space_type>			_o_q;
	qi::rule <siter, Token *(), qi::space_type>			_o_r;
	qi::rule <siter, Token *(), qi::space_type>			_o_cz;
	qi::rule <siter, Token *(), qi::space_type>			_o_cq;
	qi::rule <siter, Token *(), qi::space_type>			_o_cr;
	
	qi::rule <siter, Token *(), qi::space_type>			_o_vz;
	qi::rule <siter, Token *(), qi::space_type>			_o_vq;
	qi::rule <siter, Token *(), qi::space_type>			_o_vr;
	qi::rule <siter, Token *(), qi::space_type>			_o_vgq;
	qi::rule <siter, Token *(), qi::space_type>			_o_vgr;
	qi::rule <siter, Token *(), qi::space_type>			_o_vcz;
	qi::rule <siter, Token *(), qi::space_type>			_o_vcq;
	qi::rule <siter, Token *(), qi::space_type>			_o_vcr;
	qi::rule <siter, Token *(), qi::space_type>			_o_vcgq;
	qi::rule <siter, Token *(), qi::space_type>			_o_vcgr;
	
	qi::rule <siter, Token *(), qi::space_type>			_o_mz;
	qi::rule <siter, Token *(), qi::space_type>			_o_mq;
	qi::rule <siter, Token *(), qi::space_type>			_o_mgr;
	qi::rule <siter, Token *(), qi::space_type>			_o_mgq;
	qi::rule <siter, Token *(), qi::space_type>			_o_mr;
	qi::rule <siter, Token *(), qi::space_type>			_o_mcz;
	qi::rule <siter, Token *(), qi::space_type>			_o_mcq;
	qi::rule <siter, Token *(), qi::space_type>			_o_mcr;
	qi::rule <siter, Token *(), qi::space_type>			_o_mcgq;
	qi::rule <siter, Token *(), qi::space_type>			_o_mcgr;
	
	// Type parsers
	qi::rule <siter, std::string ()>					_str;

	qi::rule <siter, Z (), qi::space_type>					_z;
	qi::rule <siter, Q (), qi::space_type>					_q;
	qi::rule <siter, R (), qi::space_type>					_r;
	qi::rule <siter, Q (), qi::space_type>					_gq;
	qi::rule <siter, R (), qi::space_type>					_gr;
	qi::rule <siter, CZ (), qi::space_type>					_cz;
	qi::rule <siter, CQ (), qi::space_type>					_cq;
	qi::rule <siter, CR (), qi::space_type>					_cr;
	qi::rule <siter, CQ (), qi::space_type>					_cgq;
	qi::rule <siter, CR (), qi::space_type>					_cgr;
	
	qi::rule <siter, std::vector <Z> (), qi::space_type>			_vz;
	qi::rule <siter, std::vector <Q> (), qi::space_type>			_vq;
	qi::rule <siter, std::vector <R> (), qi::space_type>			_vr;
	qi::rule <siter, std::vector <Q> (), qi::space_type>			_vgq;
	qi::rule <siter, std::vector <R> (), qi::space_type>			_vgr;
	qi::rule <siter, std::vector <CZ> (), qi::space_type>			_vcz;
	qi::rule <siter, std::vector <CQ> (), qi::space_type>			_vcq;
	qi::rule <siter, std::vector <CR> (), qi::space_type>			_vcr;
	qi::rule <siter, std::vector <CQ> (), qi::space_type>			_vcgq;
	qi::rule <siter, std::vector <CR> (), qi::space_type>			_vcgr;
	
	qi::rule <siter, std::vector <std::vector <Z>> (), qi::space_type>	_mz;
	qi::rule <siter, std::vector <std::vector <Q>> (), qi::space_type>	_mq;
	qi::rule <siter, std::vector <std::vector <R>> (), qi::space_type>	_mr;
	qi::rule <siter, std::vector <std::vector <Q>> (), qi::space_type>	_mgq;
	qi::rule <siter, std::vector <std::vector <R>> (), qi::space_type>	_mgr;
	qi::rule <siter, std::vector <std::vector <CZ>> (), qi::space_type>	_mcz;
	qi::rule <siter, std::vector <std::vector <CQ>> (), qi::space_type>	_mcq;
	qi::rule <siter, std::vector <std::vector <CR>> (), qi::space_type>	_mcr;
	qi::rule <siter, std::vector <std::vector <CQ>> (), qi::space_type>	_mcgq;
	qi::rule <siter, std::vector <std::vector <CR>> (), qi::space_type>	_mcgr;

	// Vector and matrix intermediates
	qi::rule <siter, std::vector <Z> (), qi::space_type>			_vz_inter;
	qi::rule <siter, std::vector <Q> (), qi::space_type>			_vq_inter;
	qi::rule <siter, std::vector <R> (), qi::space_type>			_vr_inter;
	qi::rule <siter, std::vector <Q> (), qi::space_type>			_vgq_inter;
	qi::rule <siter, std::vector <R> (), qi::space_type>			_vgr_inter;
	qi::rule <siter, std::vector <CZ> (), qi::space_type>			_vcz_inter;
	qi::rule <siter, std::vector <CQ> (), qi::space_type>			_vcq_inter;
	qi::rule <siter, std::vector <CR> (), qi::space_type>			_vcr_inter;
	qi::rule <siter, std::vector <CQ> (), qi::space_type>			_vcgq_inter;
	qi::rule <siter, std::vector <CR> (), qi::space_type>			_vcgr_inter;
	
	qi::rule <siter, std::vector <std::vector <Z>> (), qi::space_type>	_mz_inter;
	qi::rule <siter, std::vector <std::vector <Q>> (), qi::space_type>	_mq_inter;
	qi::rule <siter, std::vector <std::vector <R>> (), qi::space_type>	_mr_inter;
	qi::rule <siter, std::vector <std::vector <Q>> (), qi::space_type>	_mgq_inter;
	qi::rule <siter, std::vector <std::vector <R>> (), qi::space_type>	_mgr_inter;
	qi::rule <siter, std::vector <std::vector <CZ>> (), qi::space_type>	_mcz_inter;
	qi::rule <siter, std::vector <std::vector <CQ>> (), qi::space_type>	_mcq_inter;
	qi::rule <siter, std::vector <std::vector <CR>> (), qi::space_type>	_mcr_inter;
	qi::rule <siter, std::vector <std::vector <CQ>> (), qi::space_type>	_mcgq_inter;
	qi::rule <siter, std::vector <std::vector <CR>> (), qi::space_type>	_mcgr_inter;

};

}

#endif
