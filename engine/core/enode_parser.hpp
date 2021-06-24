#ifndef ENODE_PARSER_H_
#define ENODE_PARSER_H_

// C/C++ headers
#include <fstream>
#include <stdexcept>
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
#include <boost/spirit/home/support/common_terminals.hpp>

// Engine headers
#include "enode.hpp"
#include "primitive.hpp"

namespace zhetapi {

// Parser
class EnodeParser : public boost::spirit::qi::grammar
		      <std::string::const_iterator,
		      Enode (),
		      boost::spirit::qi::space_type> {
public:
	// Aliases
	template <class A, class B, class C>
	using Rule = boost::spirit::qi::rule <A, B, C>;

	using Space = boost::spirit::qi::space_type;

	template <class A, class B>
	using Symbols = boost::spirit::qi::symbols <A, B>;

	using Siter = std::string::const_iterator;

	// Add alilas for rule <siter, ...>
private:
	// Primitive types
	Rule <Siter, Primitive (), Space>	_int;
	Rule <Siter, Primitive (), Space>	_double;

	// Operations
	Rule <Siter, OpCode, Space>		_addition;
	Rule <Siter, OpCode, Space>		_subtraction;
	Rule <Siter, OpCode, Space>		_multiplication;
	Rule <Siter, OpCode, Space>		_division;
	
	// Precedences
	Rule <Siter, OpCode, Space>		_t0_bin;
	Rule <Siter, OpCode, Space>		_t1_bin;
	
	// Structural
	Rule <Siter, Primitive (), Space>	_primop;
	Rule <Siter, Enode (), Space>		_term;
	
	Rule <Siter, Enode (), Space>		_start;

	// Miscellaneous
	Symbols <char const, char const>	_esc; // Escape characters
public:
	EnodeParser();

	static EnodeParser eparser;

	// Exceptions
	class bad_parse : public std::runtime_error {
	public:
		bad_parse(const std::string &str)
				: std::runtime_error("Error parsing \""
						+ str + "\"") {}
	};
};

// Construction function
Enode strmake(const std::string &);

}

#endif
