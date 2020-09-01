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

#include <iostream>
#include <string>
#include <vector>
#include <complex>

#include <complex.hpp>
#include <node.hpp>

using namespace std;

using namespace boost;

using namespace boost::fusion;
using namespace boost::spirit;
using namespace boost::phoenix;

using namespace ascii;

typedef std::string::const_iterator siter;

// Parser
struct complex_parser : qi::grammar <siter, Complex <double> ()> {

	complex_parser() : complex_parser::base_type(start) {
		cmpl = double_[_val = phoenix::construct <Complex <double>> (0, _1)] >> 'i';
		real = double_;

		start = (cmpl | real) [_val = _1];
	}

	qi::rule <siter, Complex <double> ()> start;

	qi::rule <siter, Complex <double> ()> real;
	qi::rule <siter, Complex <double> ()> cmpl;
};

// Parser
struct parser : qi::grammar <siter, zhetapi::node ()> {

	parser() : parser::base_type(start) {
		o_z = int_ [
			_val = phoenix::new_ <zhetapi::operand <int>> (_1)
		];

		n_o_z = o_z [
			_val = phoenix::construct <zhetapi::node> (_1, std::vector <zhetapi::node> {})
		];

		start = (n_o_z) [_val = _1];
	}

	qi::rule <siter, zhetapi::node ()> start;

	qi::rule <siter, zhetapi::token *()> o_z;

	qi::rule <siter, zhetapi::node ()> n_o_z;
};

///////////////////////////////////////////////////////////////////////////////
//  Main program
///////////////////////////////////////////////////////////////////////////////
int main()
{
	complex_parser cmplx; // Our grammar

	std::string str = "8i";

	Complex <double> result;

	siter iter = str.begin();
	siter end = str.end();

	bool r = parse(iter, end, cmplx, result);
	
	std::cout << "result = " << result << std::endl;

	// nodes
	parser prs;

	str = "81";

	zhetapi::node nd;
	
	iter = str.begin();
	end = str.end();

	r = parse(iter, end, prs, nd);

	cout << "nd:" << endl;
	nd.print();

	return 0;
}
