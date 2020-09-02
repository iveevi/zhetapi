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

// C/C++ headers
#include <complex>
#include <iostream>
#include <string>
#include <vector>

// Engine headers
#include <complex.hpp>
#include <node.hpp>
#include <rational.hpp>

using namespace std;

using namespace boost;

using namespace boost::fusion;
using namespace boost::spirit;
using namespace boost::phoenix;

typedef std::string::const_iterator siter;

// Parser
template <class T, class U>
struct parser : qi::grammar <siter, zhetapi::node ()> {

	__TYPEDEFS__

	parser() : parser::base_type(__start) {
		// Type parsers (change dependence on int_ and double_ parsers
		// specifically)
		__z = int_;

		__q = (int_ >> '/' >> int_) [
			_val = phoenix::construct <Q> (_1, _2)
		];

		__r = double_;

		__cz = (__z >> 'i') [
			_val = phoenix::construct <CZ> (0, _1)
		];
		
		__cq = (__q >> 'i') [
			_val = phoenix::construct <CQ> (0, _1)
		];
		
		__cr = (__r >> 'i') [
			_val = phoenix::construct <CR> (0, _1)
		];

		// Token parsers
		__o_z = __z [
			_val = phoenix::new_ <zhetapi::operand <Z>> (_1)
		];
		
		__o_q = __q [
			_val = phoenix::new_ <zhetapi::operand <Q>> (_1)
		];

		__o_r = __r [
			_val = phoenix::new_ <zhetapi::operand <R>> (_1)
		];
		
		__o_cz = __cz [
			_val = phoenix::new_ <zhetapi::operand <CZ>> (_1)
		];
		
		__o_cq = __cq [
			_val = phoenix::new_ <zhetapi::operand <CQ>> (_1)
		];

		__o_cr = __cr [
			_val = phoenix::new_ <zhetapi::operand <CR>> (_1)
		];

		// Nodes
		__node_opd = (
				__o_cq
				| __o_cr
				| __o_cz
				| __o_q
				| __o_r
				| __o_z
			) [
			_val = phoenix::construct <zhetapi::node> (_1,
					std::vector <zhetapi::node> {})
		];

		__start = (__node_opd) [_val = _1];
	}

	qi::rule <siter, zhetapi::node ()>	__start;

	// Token parsers
	qi::rule <siter, zhetapi::token *()>	__o_z;
	qi::rule <siter, zhetapi::token *()>	__o_q;
	qi::rule <siter, zhetapi::token *()>	__o_r;
	qi::rule <siter, zhetapi::token *()>	__o_cz;
	qi::rule <siter, zhetapi::token *()>	__o_cq;
	qi::rule <siter, zhetapi::token *()>	__o_cr;
	
	// Type parsers
	qi::rule <siter, Z ()>			__z;
	qi::rule <siter, Q ()>			__q;
	qi::rule <siter, R>			__r;
	qi::rule <siter, CZ ()>			__cz;
	qi::rule <siter, CQ>			__cq;
	qi::rule <siter, CR>			__cr;

	qi::rule <siter, zhetapi::node ()>	__node_opd;
};

///////////////////////////////////////////////////////////////////////////////
//  Main program
///////////////////////////////////////////////////////////////////////////////
int main()
{
	// nodes
	parser <double, int> pars;

	std::string str;

	siter iter;
	siter end;

	bool r;

	while (getline(cin, str)) {
		cout << "-------------" << endl;

		zhetapi::node nd;
		
		iter = str.begin();
		end = str.end();

		r = parse(iter, end, pars, nd);

		cout << "str: " << str << endl;

		if (r) {
			// Status
			cout << "Parsing succeeded";

			if (iter != end)
				cout << " (NOT FULLY PARSED)";

			cout << endl;

			// Node
			cout << "nd:" << endl;
			nd.print();
		} else {
			cout << "Parsing failed" << endl;
		}
	}

	return 0;
}
