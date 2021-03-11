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

#define __add_operation_symbol(name, str)				\
	name = lit(#str) [						\
		_val = phoenix::new_					\
			<operation_holder> (std::string(#str))		\
	];

#define __add_operation_heter_symbol(name, str, act)			\
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
struct parser : qi::grammar <siter, zhetapi::node (), qi::space_type> {

	parser();

	qi::symbols <char const, char const>					__esc;
	
	qi::rule <siter, zhetapi::node (), qi::space_type>			__start;
	
	// Nodes
	qi::rule <siter, zhetapi::node (), qi::space_type>			__node_expr;
	qi::rule <siter, zhetapi::node (), qi::space_type>			__node_term;
	qi::rule <siter, zhetapi::node (), qi::space_type>			__node_factor;
	qi::rule <siter, zhetapi::node (), qi::space_type>			__node_prep;
	qi::rule <siter, zhetapi::node (), qi::space_type>			__node_rept;
	qi::rule <siter, zhetapi::node (), qi::space_type>			__node_prth;
	qi::rule <siter, zhetapi::node (), qi::space_type>			__node_var;
	qi::rule <siter, zhetapi::node (), qi::space_type>			__node_opd;

	// Parameter pack
	qi::rule <siter, ::std::vector <zhetapi::node> (), qi::space_type>	__node_pack;

	// Identifiers
	qi::rule <siter, ::std::string ()>					__ident;

	// Operations
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__t0_bin;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__t1_bin;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__t2_bin;

	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__t_pre;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__t_post;

	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__eq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__neq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__ge;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__le;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__leq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__geq;

	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__post_incr;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__post_decr;
	
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__pre_incr;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__pre_decr;

	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__plus;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__minus;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__dot;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__mod;
	
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__times;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__divide;
	
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__power;

	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__collection;

	// Operands

	// Token parsers
	qi::rule <siter, zhetapi::Token *()>					__o_str;

	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_z;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_q;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_r;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_cz;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_cq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_cr;
	
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vz;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vr;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vgq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vgr;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vcz;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vcq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vcr;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vcgq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_vcgr;
	
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mz;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mgr;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mgq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mr;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mcz;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mcq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mcr;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mcgq;
	qi::rule <siter, zhetapi::Token *(), qi::space_type>			__o_mcgr;
	
	// Type parsers
	qi::rule <siter, ::std::string ()>					__str;

	qi::rule <siter, Z (), qi::space_type>					__z;
	qi::rule <siter, Q (), qi::space_type>					__q;
	qi::rule <siter, R (), qi::space_type>					__r;
	qi::rule <siter, Q (), qi::space_type>					__gq;
	qi::rule <siter, R (), qi::space_type>					__gr;
	qi::rule <siter, CZ (), qi::space_type>					__cz;
	qi::rule <siter, CQ (), qi::space_type>					__cq;
	qi::rule <siter, CR (), qi::space_type>					__cr;
	qi::rule <siter, CQ (), qi::space_type>					__cgq;
	qi::rule <siter, CR (), qi::space_type>					__cgr;
	
	qi::rule <siter, ::std::vector <Z> (), qi::space_type>			__vz;
	qi::rule <siter, ::std::vector <Q> (), qi::space_type>			__vq;
	qi::rule <siter, ::std::vector <R> (), qi::space_type>			__vr;
	qi::rule <siter, ::std::vector <Q> (), qi::space_type>			__vgq;
	qi::rule <siter, ::std::vector <R> (), qi::space_type>			__vgr;
	qi::rule <siter, ::std::vector <CZ> (), qi::space_type>			__vcz;
	qi::rule <siter, ::std::vector <CQ> (), qi::space_type>			__vcq;
	qi::rule <siter, ::std::vector <CR> (), qi::space_type>			__vcr;
	qi::rule <siter, ::std::vector <CQ> (), qi::space_type>			__vcgq;
	qi::rule <siter, ::std::vector <CR> (), qi::space_type>			__vcgr;
	
	qi::rule <siter, ::std::vector <::std::vector <Z>> (), qi::space_type>	__mz;
	qi::rule <siter, ::std::vector <::std::vector <Q>> (), qi::space_type>	__mq;
	qi::rule <siter, ::std::vector <::std::vector <R>> (), qi::space_type>	__mr;
	qi::rule <siter, ::std::vector <::std::vector <Q>> (), qi::space_type>	__mgq;
	qi::rule <siter, ::std::vector <::std::vector <R>> (), qi::space_type>	__mgr;
	qi::rule <siter, ::std::vector <::std::vector <CZ>> (), qi::space_type>	__mcz;
	qi::rule <siter, ::std::vector <::std::vector <CQ>> (), qi::space_type>	__mcq;
	qi::rule <siter, ::std::vector <::std::vector <CR>> (), qi::space_type>	__mcr;
	qi::rule <siter, ::std::vector <::std::vector <CQ>> (), qi::space_type>	__mcgq;
	qi::rule <siter, ::std::vector <::std::vector <CR>> (), qi::space_type>	__mcgr;

	// Vector and matrix intermediates
	qi::rule <siter, ::std::vector <Z> (), qi::space_type>			__vz_inter;
	qi::rule <siter, ::std::vector <Q> (), qi::space_type>			__vq_inter;
	qi::rule <siter, ::std::vector <R> (), qi::space_type>			__vr_inter;
	qi::rule <siter, ::std::vector <Q> (), qi::space_type>			__vgq_inter;
	qi::rule <siter, ::std::vector <R> (), qi::space_type>			__vgr_inter;
	qi::rule <siter, ::std::vector <CZ> (), qi::space_type>			__vcz_inter;
	qi::rule <siter, ::std::vector <CQ> (), qi::space_type>			__vcq_inter;
	qi::rule <siter, ::std::vector <CR> (), qi::space_type>			__vcr_inter;
	qi::rule <siter, ::std::vector <CQ> (), qi::space_type>			__vcgq_inter;
	qi::rule <siter, ::std::vector <CR> (), qi::space_type>			__vcgr_inter;
	
	qi::rule <siter, ::std::vector <::std::vector <Z>> (), qi::space_type>	__mz_inter;
	qi::rule <siter, ::std::vector <::std::vector <Q>> (), qi::space_type>	__mq_inter;
	qi::rule <siter, ::std::vector <::std::vector <R>> (), qi::space_type>	__mr_inter;
	qi::rule <siter, ::std::vector <::std::vector <Q>> (), qi::space_type>	__mgq_inter;
	qi::rule <siter, ::std::vector <::std::vector <R>> (), qi::space_type>	__mgr_inter;
	qi::rule <siter, ::std::vector <::std::vector <CZ>> (), qi::space_type>	__mcz_inter;
	qi::rule <siter, ::std::vector <::std::vector <CQ>> (), qi::space_type>	__mcq_inter;
	qi::rule <siter, ::std::vector <::std::vector <CR>> (), qi::space_type>	__mcr_inter;
	qi::rule <siter, ::std::vector <::std::vector <CQ>> (), qi::space_type>	__mcgq_inter;
	qi::rule <siter, ::std::vector <::std::vector <CR>> (), qi::space_type>	__mcgr_inter;

};

}

#endif
