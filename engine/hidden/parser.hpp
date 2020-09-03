// C/C++ headers
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
#include <node.hpp>

namespace zhetapi {

	typedef std::string::const_iterator siter;

	using namespace boost;

	using namespace boost::spirit;
	using namespace boost::phoenix;

	typedef ascii::blank_type skip;

	// Parser
	template <class T, class U>
	struct parser : qi::grammar <siter, zhetapi::node ()> {

		__TYPEDEFS__

		parser() : parser::base_type(__start) {
			// Type parsers (change dependence on int_ and double_ parsers
			// specifically)

			// Reals
			__z = int_;

			__q = (int_ >> '/' >> int_) [
				_val = phoenix::construct <Q> (_1, _2)
			];

			// __r = double_;
			__r = qi::real_parser <R, qi::strict_real_policies <R>> ();

			// Complex
			__cz = (__z >> 'i') [
				_val = phoenix::construct <CZ> (0, _1)
			];
			
			__cq = (__q >> 'i') [
				_val = phoenix::construct <CQ> (0, _1)
			];
			
			__cr = (__r >> 'i') [
				_val = phoenix::construct <CR> (0, _1)
			];

			// Vector

			__vz_inter = __z % ',';
			__vq_inter = __q % ',';
			__vr_inter = __r % ',';
			__vcz_inter = __cz % ',';
			__vcq_inter = __cq % ',';
			__vcr_inter = __cr % ',';
			
			__vz = ('[' >> __vz_inter >> ']') [
				_val = _1
			];
			
			__vq = ('[' >> __vq_inter >> ']') [
				_val = _1
			];
			
			__vr = ('[' >> __vr_inter >> ']') [
				_val = _1
			];
			
			__vcz = ('[' >> __vcz_inter >> ']') [
				_val = _1
			];
			
			__vcq = ('[' >> __vcq_inter >> ']') [
				_val = _1
			];
			
			__vcr = ('[' >> __vcr_inter >> ']') [
				_val = _1
			];

			// Matrix
			
			__mz_inter = __vz % ',';
			__mq_inter = __vq % ',';
			__mr_inter = __vr % ',';
			__mcz_inter = __vcz % ',';
			__mcq_inter = __vcq % ',';
			__mcr_inter = __vcr % ',';
			
			__mz = ('[' >> __mz_inter >> ']') [
				_val = _1
			];
			
			__mq = ('[' >> __mq_inter >> ']') [
				_val = _1
			];
			
			__mr = ('[' >> __mr_inter >> ']') [
				_val = _1
			];
			
			__mcz = ('[' >> __mcz_inter >> ']') [
				_val = _1
			];
			
			__mcq = ('[' >> __mcq_inter >> ']') [
				_val = _1
			];
			
			__mcr = ('[' >> __mcr_inter >> ']') [
				_val = _1
			];
			
			// Token parsers

			// Reals
			__o_z = __z [
				_val = phoenix::new_ <zhetapi::operand <Z>> (_1)
			];
			
			__o_q = __q [
				_val = phoenix::new_ <zhetapi::operand <Q>> (_1)
			];

			__o_r = __r [
				_val = phoenix::new_ <zhetapi::operand <R>> (_1)
			];
			
			// Complex
			__o_cz = __cz [
				_val = phoenix::new_ <zhetapi::operand <CZ>> (_1)
			];
			
			__o_cq = __cq [
				_val = phoenix::new_ <zhetapi::operand <CQ>> (_1)
			];

			__o_cr = __cr [
				_val = phoenix::new_ <zhetapi::operand <CR>> (_1)
			];

			// Vector
			__o_vz = __vz [
				_val = phoenix::new_ <zhetapi::operand <VZ>> (_1)
			];
			
			__o_vq = __vq [
				_val = phoenix::new_ <zhetapi::operand <VQ>> (_1)
			];
			
			__o_vr = __vr [
				_val = phoenix::new_ <zhetapi::operand <VR>> (_1)
			];
			
			__o_vcz = __vcz [
				_val = phoenix::new_ <zhetapi::operand <VCZ>> (_1)
			];
			
			__o_vcq = __vcq [
				_val = phoenix::new_ <zhetapi::operand <VCQ>> (_1)
			];
			
			__o_vcr = __vcr [
				_val = phoenix::new_ <zhetapi::operand <VCR>> (_1)
			];

			// Matrix
			__o_mz = __mz [
				_val = phoenix::new_ <zhetapi::operand <MZ>> (_1)
			];
			
			__o_mq = __mq [
				_val = phoenix::new_ <zhetapi::operand <MQ>> (_1)
			];
			
			__o_mr = __mr [
				_val = phoenix::new_ <zhetapi::operand <MR>> (_1)
			];
			
			__o_mcz = __mcz [
				_val = phoenix::new_ <zhetapi::operand <MCZ>> (_1)
			];
			
			__o_mcq = __mcq [
				_val = phoenix::new_ <zhetapi::operand <MCQ>> (_1)
			];
			
			__o_mcr = __mcr [
				_val = phoenix::new_ <zhetapi::operand <MCR>> (_1)
			];

			// Nodes
			__node_opd = (
					__o_cq
					| __o_cr
					| __o_cz

					| __o_q
					| __o_r
					
					| __o_vcq
					| __o_vcr
					| __o_vcz

					| __o_vq
					| __o_vr
					| __o_vz
					| __o_z
					
					| __o_mcq
					| __o_mcr
					| __o_mcz

					| __o_mq
					| __o_mr
					| __o_mz
				) [
				_val = phoenix::construct <zhetapi::node> (_1,
						std::vector <zhetapi::node> {})
			];

			__start = (__node_opd) [_val = _1];

			// Naming rules
			__start.name("start");
			
			__o_z.name("integer operand");
			__o_q.name("rational operand");
			__o_r.name("real operand");
			__o_cz.name("complex integer operand");
			__o_cq.name("complex rational operand");
			__o_cr.name("complex real operand");
			
			__o_vz.name("vector integer operand");
			__o_vq.name("vector rational operand");
			__o_vr.name("vector real operand");
			__o_vcz.name("vector complex integer operand");
			__o_vcq.name("vector complex rational operand");
			__o_vcr.name("vector complex real operand");
			
			__o_mz.name("matrix integer operand");
			__o_mq.name("matrix rational operand");
			__o_mr.name("matrix real operand");
			__o_mcz.name("matrix complex integer operand");
			__o_mcq.name("matrix complex rational operand");
			__o_mcr.name("matrix complex real operand");

			__z.name("integer");
			__q.name("rational");
			__r.name("real");
			__cz.name("complex integer");
			__cq.name("complex rational");
			__cr.name("complex real");
			
			__vz.name("vector integer");
			__vq.name("vector rational");
			__vr.name("vector real");
			__vcz.name("vector complex integer");
			__vcq.name("vector complex rational");
			__vcr.name("vector complex real");
			
			__mz.name("matrix integer");
			__mq.name("matrix rational");
			__mr.name("matrix real");
			__mcz.name("matrix complex integer");
			__mcq.name("matrix complex rational");
			__mcr.name("matrix complex real");
			
			__vz_inter.name("intermediate vector integer");
			__vq_inter.name("intermediate vector rational");
			__vr_inter.name("intermediate vector real");
			__vcz_inter.name("intermediate vector complex integer");
			__vcq_inter.name("intermediate vector complex rational");
			__vcr_inter.name("intermediate vector complex real");
			
			__mz_inter.name("intermediate matrix integer");
			__mq_inter.name("intermediate matrix rational");
			__mr_inter.name("intermediate matrix real");
			__mcz_inter.name("intermediate matrix complex integer");
			__mcq_inter.name("intermediate matrix complex rational");
			__mcr_inter.name("intermediate matrix complex real");

			__node_opd.name("node operand");

			// Debug

#ifdef DEBUG_PARSER
			debug(__start);

			debug(__o_z);
			debug(__o_q);
			debug(__o_r);
			debug(__o_cz);
			debug(__o_cq);
			debug(__o_cr);

			debug(__o_vz);
			debug(__o_vq);
			debug(__o_vr);
			debug(__o_vcz);
			debug(__o_vcq);
			debug(__o_vcr);

			debug(__o_mz);
			debug(__o_mq);
			debug(__o_mr);
			debug(__o_mz);
			debug(__o_mcq);
			debug(__o_mcr);
			
			debug(__z);
			debug(__q);
			debug(__r);
			debug(__cz);
			debug(__cq);
			debug(__cr);

			debug(__vz);
			debug(__vq);
			debug(__vr);
			debug(__vcz);
			debug(__vcq);
			debug(__vcr);

			debug(__mz);
			debug(__mq);
			debug(__mr);
			debug(__mz);
			debug(__mcq);
			debug(__mcr);

			debug(__node_opd);
#endif

		}

		qi::rule <siter, zhetapi::node ()>			__start;

		// Token parsers
		qi::rule <siter, zhetapi::token *()>			__o_z;
		qi::rule <siter, zhetapi::token *()>			__o_q;
		qi::rule <siter, zhetapi::token *()>			__o_r;
		qi::rule <siter, zhetapi::token *()>			__o_cz;
		qi::rule <siter, zhetapi::token *()>			__o_cq;
		qi::rule <siter, zhetapi::token *()>			__o_cr;
		
		qi::rule <siter, zhetapi::token *()>			__o_vz;
		qi::rule <siter, zhetapi::token *()>			__o_vq;
		qi::rule <siter, zhetapi::token *()>			__o_vr;
		qi::rule <siter, zhetapi::token *()>			__o_vcz;
		qi::rule <siter, zhetapi::token *()>			__o_vcq;
		qi::rule <siter, zhetapi::token *()>			__o_vcr;
		
		qi::rule <siter, zhetapi::token *()>			__o_mz;
		qi::rule <siter, zhetapi::token *()>			__o_mq;
		qi::rule <siter, zhetapi::token *()>			__o_mr;
		qi::rule <siter, zhetapi::token *()>			__o_mcz;
		qi::rule <siter, zhetapi::token *()>			__o_mcq;
		qi::rule <siter, zhetapi::token *()>			__o_mcr;
		
		// Type parsers
		qi::rule <siter, Z ()>					__z;
		qi::rule <siter, Q ()>					__q;
		qi::rule <siter, R ()>					__r;
		qi::rule <siter, CZ ()>					__cz;
		qi::rule <siter, CQ ()>					__cq;
		qi::rule <siter, CR ()>					__cr;
		
		qi::rule <siter, std::vector <Z> ()>			__vz;
		qi::rule <siter, std::vector <Q> ()>			__vq;
		qi::rule <siter, std::vector <R> ()>			__vr;
		qi::rule <siter, std::vector <CZ> ()>			__vcz;
		qi::rule <siter, std::vector <CQ> ()>			__vcq;
		qi::rule <siter, std::vector <CR> ()>			__vcr;
		
		qi::rule <siter, std::vector <std::vector <Z>> ()>	__mz;
		qi::rule <siter, std::vector <std::vector <Q>> ()>	__mq;
		qi::rule <siter, std::vector <std::vector <R>> ()>	__mr;
		qi::rule <siter, std::vector <std::vector <CZ>> ()>	__mcz;
		qi::rule <siter, std::vector <std::vector <CQ>> ()>	__mcq;
		qi::rule <siter, std::vector <std::vector <CR>> ()>	__mcr;

		// Vector and matrix intermediates
		qi::rule <siter, std::vector <Z> ()>			__vz_inter;
		qi::rule <siter, std::vector <Q> ()>			__vq_inter;
		qi::rule <siter, std::vector <R> ()>			__vr_inter;
		qi::rule <siter, std::vector <CZ> ()>			__vcz_inter;
		qi::rule <siter, std::vector <CQ> ()>			__vcq_inter;
		qi::rule <siter, std::vector <CR> ()>			__vcr_inter;
		
		qi::rule <siter, std::vector <std::vector <Z>> ()>	__mz_inter;
		qi::rule <siter, std::vector <std::vector <Q>> ()>	__mq_inter;
		qi::rule <siter, std::vector <std::vector <R>> ()>	__mr_inter;
		qi::rule <siter, std::vector <std::vector <CZ>> ()>	__mcz_inter;
		qi::rule <siter, std::vector <std::vector <CQ>> ()>	__mcq_inter;
		qi::rule <siter, std::vector <std::vector <CR>> ()>	__mcr_inter;
		

		// Nodes
		qi::rule <siter, zhetapi::node ()>			__node_opd;
	};

}
