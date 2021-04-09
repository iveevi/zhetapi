#include <core/parser.hpp>

namespace zhetapi {

parser::parser() : parser::base_type(_start)
{
	_esc.add("\\a", '\a')("\\b", '\b')("\\f", '\f')("\\n", '\n')
		("\\r", '\r')("\\t", '\t')("\\v", '\v')("\\\\", '\\')
		("\\\'", '\'')("\\\"", '\"');

	/*
	 * Parser for an identifier. Used to construct variable
	 * clusters.
	 */
	_ident = +qi::char_("a-zA-Z$_");

	_str = +(_esc | (qi::char_ - '\"'));

	// Operation parsers
	_add_operation_symbol(_plus, +);
	_add_operation_symbol(_minus, -);
	_add_operation_symbol(_times, *);
	_add_operation_symbol(_divide, /);
	_add_operation_symbol(_power, ^);
	_add_operation_symbol(_dot, @);
	_add_operation_symbol(_mod, %);
	
	// Binary comparison
	_add_operation_symbol(_eq, ==);
	_add_operation_symbol(_neq, !=);
	_add_operation_symbol(_ge, >);
	_add_operation_symbol(_le, <);
	_add_operation_symbol(_geq, >=);
	_add_operation_symbol(_leq, <=);

	// Unary increment/decrement
	_add_operation_heter_symbol(_post_incr, ++, p++);
	_add_operation_heter_symbol(_post_decr, --, p--);
	
	_add_operation_heter_symbol(_pre_incr, ++, r++);
	_add_operation_heter_symbol(_pre_decr, --, r--);

	// Miscellaenous
	_add_operation_symbol(_attribute, .);

	/*
	 * Represents a binary operation of lowest priotrity.
	 * These are used to combine terms into expressions.
	 * Exmaples of such operations are addition,
	 * subtraction and the dot product.
	 */
	_t0_bin = _plus | _minus | _dot | _mod
			| _eq | _neq | _geq
			| _leq | _ge | _le;

	/*
	 * Represents a binary operation of second to lowest
	 * priority. Connects factors into term. Examples are
	 * multiplication and division.
	 */
	_t1_bin = _times | _divide;
	
	/*
	 * Represents other binary operations with precedence
	 * higher than that of _t1_bin. Examples are the
	 * exponentiation operation.
	 */
	_t2_bin = _attribute | _power;

	_t_post = _post_incr | _post_decr;
	_t_pre = _pre_incr | _pre_decr;

	// Type parsers (change dependence on int_ and double_ parsers
	// specifically)

	// Reals
	_z = int_;

	_q = (int_ >> '/' >> int_) [
		_val = phoenix::construct <Q> (_1, _2)
	];

	// _r = double_;
	_r = qi::real_parser <R, qi::strict_real_policies <R>> ();

	// Generalized
	_gq = _q | _z;
	_gr = _r | _gq;
	
	// Complex
	_cz = (_z >> 'i') [
		_val = phoenix::construct <CZ> (0, _1)
	];
	
	_cq = (_q >> 'i') [
		_val = phoenix::construct <CQ> (0, _1)
	];
	
	_cr = (_r >> 'i') [
		_val = phoenix::construct <CR> (0, _1)
	];
	
	_cgq = (_gq >> 'i') [
		_val = phoenix::construct <CQ> (0, _1)
	];
	
	_cgr = (_gr >> 'i') [
		_val = phoenix::construct <CR> (0, _1)
	];

	// Vector

	_vz_inter = _z % ',';
	_vq_inter = _q % ',';
	_vr_inter = _r % ',';
	_vgq_inter = _gq % ',';
	_vgr_inter = _gr % ',';
	_vcz_inter = _cz % ',';
	_vcq_inter = _cq % ',';
	_vcr_inter = _cr % ',';
	_vcgq_inter = _cgq % ',';
	_vcgr_inter = _cgr % ',';
	
	_vz = ('[' >> _vz_inter >> ']') [
		_val = _1
	];
	
	_vq = ('[' >> _vq_inter >> ']') [
		_val = _1
	];
	
	_vr = ('[' >> _vr_inter >> ']') [
		_val = _1
	];
	
	_vgq = ('[' >> _vgq_inter >> ']') [
		_val = _1
	];
	
	_vgr = ('[' >> _vgr_inter >> ']') [
		_val = _1
	];
	
	_vcz = ('[' >> _vcz_inter >> ']') [
		_val = _1
	];
	
	_vcq = ('[' >> _vcq_inter >> ']') [
		_val = _1
	];
	
	_vcr = ('[' >> _vcr_inter >> ']') [
		_val = _1
	];
	
	_vcgq = ('[' >> _vcgq_inter >> ']') [
		_val = _1
	];
	
	_vcgr = ('[' >> _vcgr_inter >> ']') [
		_val = _1
	];

	// Matrix
	_mz_inter = _vz % ',';
	_mq_inter = _vq % ',';
	_mr_inter = _vr % ',';
	_mgq_inter = _vgq % ',';
	_mgr_inter = _vgr % ',';
	_mcz_inter = _vcz % ',';
	_mcq_inter = _vcq % ',';
	_mcr_inter = _vcr % ',';
	_mcgq_inter = _vcgq % ',';
	_mcgr_inter = _vcgr % ',';
	
	_mz = ('[' >> _mz_inter >> ']') [
		_val = _1
	];
	
	_mq = ('[' >> _mq_inter >> ']') [
		_val = _1
	];
	
	_mr = ('[' >> _mr_inter >> ']') [
		_val = _1
	];
	
	_mgq = ('[' >> _mgq_inter >> ']') [
		_val = _1
	];
	
	_mgr = ('[' >> _mgr_inter >> ']') [
		_val = _1
	];
	
	_mcz = ('[' >> _mcz_inter >> ']') [
		_val = _1
	];
	
	_mcq = ('[' >> _mcq_inter >> ']') [
		_val = _1
	];
	
	_mcr = ('[' >> _mcr_inter >> ']') [
		_val = _1
	];
	
	_mcgq = ('[' >> _mcgq_inter >> ']') [
		_val = _1
	];
	
	_mcgr = ('[' >> _mcgr_inter >> ']') [
		_val = _1
	];
	
	// Token parsers

	// Reals
	_o_str = qi::lit('\"') >> _str [
		_val = phoenix::new_ <opd_s> (_1)
	] >> qi::lit('\"');

	_o_z = _z [
		_val = phoenix::new_ <opd_z> (_1)
	];
	
	_o_q = _q [
		_val = phoenix::new_ <opd_q> (_1)
	];

	_o_r = _r [
		_val = phoenix::new_ <opd_r> (_1)
	];
	
	// Complex
	_o_cz = _cz [
		_val = phoenix::new_ <opd_cz> (_1)
	];
	
	_o_cq = _cq [
		_val = phoenix::new_ <opd_cq> (_1)
	];

	_o_cr = _cr [
		_val = phoenix::new_ <opd_cr> (_1)
	];

	// Vector
	_o_vz = _vz [
		_val = phoenix::new_ <opd_v_z> (_1)
	];
	
	_o_vq = _vq [
		_val = phoenix::new_ <opd_v_q> (_1)
	];
	
	_o_vr = _vr [
		_val = phoenix::new_ <opd_v_r> (_1)
	];
	
	_o_vgq = _vgq [
		_val = phoenix::new_ <opd_v_q> (_1)
	];
	
	_o_vgr = _vgr [
		_val = phoenix::new_ <opd_v_r> (_1)
	];
	
	_o_vcz = _vcz [
		_val = phoenix::new_ <opd_v_cz> (_1)
	];
	
	_o_vcq = _vcq [
		_val = phoenix::new_ <opd_v_cq> (_1)
	];
	
	_o_vcr = _vcr [
		_val = phoenix::new_ <opd_v_cr> (_1)
	];
	
	_o_vcgq = _vcgq [
		_val = phoenix::new_ <opd_v_cq> (_1)
	];
	
	_o_vcgr = _vcgr [
		_val = phoenix::new_ <opd_v_cr> (_1)
	];

	// Matrix
	_o_mz = _mz [
		_val = phoenix::new_ <opd_m_z> (_1)
	];
	
	_o_mq = _mq [
		_val = phoenix::new_ <opd_m_q> (_1)
	];
	
	_o_mr = _mr [
		_val = phoenix::new_ <opd_m_r> (_1)
	];
	
	_o_mgq = _mgq [
		_val = phoenix::new_ <opd_m_q> (_1)
	];
	
	_o_mgr = _mgr [
		_val = phoenix::new_ <opd_m_r> (_1)
	];
	
	_o_mcz = _mcz [
		_val = phoenix::new_ <opd_m_cz> (_1)
	];
	
	_o_mcq = _mcq [
		_val = phoenix::new_ <opd_m_cq> (_1)
	];
	
	_o_mcr = _mcr [
		_val = phoenix::new_ <opd_m_cr> (_1)
	];
	
	_o_mcgq = _mcgq [
		_val = phoenix::new_ <opd_m_cq> (_1)
	];
	
	_o_mcgr = _mcgr [
		_val = phoenix::new_ <opd_m_cr> (_1)
	];

	// Nodes

	_node_pack = _start % ',';

	_collection = ('{' >> _node_pack >> '}') [
		_val = phoenix::new_ <node_list> (_1)
	];

	/*
	 * Pure numerical Operands, representing the 18
	 * primitive types of computation. The exceptions are
	 * rational numbers, which are excluded so that the
	 * order of operations can be applied. This exclusion
	 * should not cause any trouble, as integer division
	 * will yield a rational result.
	 */
	_node_opd = (
			_o_str
			| _o_cr | _o_cz
			| _o_r | _o_z
			| _o_vcr | _o_vcz
			| _o_vr | _o_vz
			| _o_vcgr
			| _o_vgr
			| _o_mcr | _o_mcz
			| _o_mr | _o_mz
			| _o_mcgr
			| _o_mgr
		) [
		_val = phoenix::construct <zhetapi::node> (_1,
				::std::vector <zhetapi::node> {})
	];

	/*
	 * A variable cluster, which is just a string of
	 * characters. The expansion/unpakcing of this variable
	 * cluster is done in the higher node_manager class,
	 * where access to the engine object is present.
	 */
	_node_var = (
			(_ident >> '(' >> _node_pack >> ')') [
				_val = phoenix::construct <zhetapi::node> (
					phoenix::new_ <variable_cluster> (_1),
					_2
				)
			]

			| _ident [_val = phoenix::construct
				<zhetapi::node> (phoenix::new_
					<variable_cluster> (_1),
					::std::vector <zhetapi::node> {})]
		);

	/*
	 * Represents a parenthesized expression.
	 */
	_node_prth = "(" >> _start [_val = _1] >> ")";

	/*
	 * Represents a repeatable factor. Examples are
	 * variables and parenthesized expressions. The reason
	 * is because terms like x3(5 + 3) are much more awkward
	 * compared to 3x(5 + 3)
	 */
	_node_rept = _node_var | _node_prth;
	
	/*
	 * Represents a series of parenthesized expression,
	 * which are mutlilpied through the use of
	 * juxtaposition.
	 */
	_node_prep = _node_rept [_val = _1] >> *(
			(_node_rept) [_val = phoenix::construct
			<zhetapi::node> (phoenix::new_
				<operation_holder>
				(::std::string("*")), _val, _1)]
		);

	/*
	 * Represents a part of a term. For example, in the term
	 * 3x, 3 and x are both collectibles.
	 */
	_node_factor = (
			_node_prep [_val = _1]

			| _node_opd [_val = _1] >> *(
				(_node_rept) [
					_val = phoenix::construct <zhetapi::node> (
						phoenix::new_ <operation_holder> (std::string("*")),
						_val,
						_1
					)
				]
			)

			| _collection [_val = _1]
		);
	
	_attr = _ident [
		_val = phoenix::construct <zhetapi::node> (
			phoenix::new_ <variable_cluster> (_1)
		)
	];

	/*
	 * Represents a term as in any mathematical expression.
	 * Should be written without addition or subtraction
	 * unless in parenthesis.
	 */
	_node_term = (
			// TODO: must rearrange attribute chains
			(_node_factor >> _power >> _node_term) [
				_val = phoenix::construct <zhetapi::node> (_2, _1, _3)
			]

			| (_node_factor >> _attribute >> _node_term) [
				_val = phoenix::construct <zhetapi::node> (_2, _1, _3)
			]
			
			| (_t_pre >> _node_var) [
				_val = phoenix::construct <zhetapi::node> (_1, _2, true)
			]

			| (_node_var >> _t_post) [
				_val = phoenix::construct <zhetapi::node> (_2, _1, true)
			]

			| _node_factor [_val = _1] >> *(
				(_t1_bin >> _node_factor)
				[_val = phoenix::construct
				<zhetapi::node> (_1, _val, _2)]
			)
		);

	/*
	 * A full expression or function definition.
	 */
	_node_expr = _node_term [_val = _1] >> *(
			(_t0_bin >> _node_term) [_val =
			phoenix::construct <zhetapi::node> (_1,
				_val, _2)]
		);
	
	// Entry point
	_start = _node_expr;

	// Naming rules
	_start.name("start");

	_node_expr.name("node expression");
	_node_term.name("node term");
	_node_opd.name("node Operand");

	_plus.name("addition");
	_minus.name("subtraction");
	_times.name("multiplication");
	_divide.name("division");
	_power.name("exponentiation");
	_attribute.name("attribute/method");
	
	_o_str.name("literal operand");

	_o_z.name("integer operand");
	_o_q.name("rational operand");
	_o_r.name("real operand");
	_o_cz.name("complex integer operand");
	_o_cq.name("complex rational operand");
	_o_cr.name("complex real operand");
	
	_o_vz.name("vector integer operand");
	_o_vq.name("vector rational operand");
	_o_vr.name("vector real operand");
	_o_vgq.name("vector general rational operand");
	_o_vgr.name("vector general real operand");
	_o_vcz.name("vector complex integer operand");
	_o_vcq.name("vector complex rational operand");
	_o_vcr.name("vector complex real operand");
	_o_vcgq.name("vector complex general rational operand");
	_o_vcgr.name("vector complex general real operand");
	
	_o_mz.name("matrix integer Operand");
	_o_mq.name("matrix rational Operand");
	_o_mr.name("matrix real Operand");
	_o_mgq.name("matrix general rational Operand");
	_o_mgr.name("matrix general real Operand");
	_o_mcz.name("matrix complex integer Operand");
	_o_mcq.name("matrix complex rational Operand");
	_o_mcr.name("matrix complex real Operand");
	_o_mcgq.name("matrix complex general rational Operand");
	_o_mcgr.name("matrix complex general real Operand");

	_str.name("literal");

	_z.name("integer");
	_q.name("rational");
	_r.name("real");
	_gq.name("general rational");
	_gr.name("general real");
	_cz.name("complex integer");
	_cq.name("complex rational");
	_cr.name("complex real");
	_cgq.name("complex general rational");
	_cgr.name("complex general real");
	
	_vz.name("vector integer");
	_vq.name("vector rational");
	_vr.name("vector real");
	_vgq.name("vector general rational");
	_vgr.name("vector general real");
	_vcz.name("vector complex integer");
	_vcq.name("vector complex rational");
	_vcr.name("vector complex real");
	_vcgq.name("vector complex general rational");
	_vcgr.name("vector complex general real");
	
	_mz.name("matrix integer");
	_mq.name("matrix rational");
	_mr.name("matrix real");
	_mgq.name("matrix general rational");
	_mgr.name("matrix general real");
	_mcz.name("matrix complex integer");
	_mcq.name("matrix complex rational");
	_mcr.name("matrix complex real");
	_mcgq.name("matrix complex general rational");
	_mcgr.name("matrix complex general real");
	
	_vz_inter.name("intermediate vector integer");
	_vq_inter.name("intermediate vector rational");
	_vr_inter.name("intermediate vector real");
	_vgq_inter.name("intermediate vector general rational");
	_vgr_inter.name("intermediate vector general real");
	_vcz_inter.name("intermediate vector complex integer");
	_vcq_inter.name("intermediate vector complex rational");
	_vcr_inter.name("intermediate vector complex real");
	_vcgq_inter.name("intermediate vector complex general rational");
	_vcgr_inter.name("intermediate vector complex general real");
	
	_mz_inter.name("intermediate matrix integer");
	_mq_inter.name("intermediate matrix rational");
	_mr_inter.name("intermediate matrix real");
	_mgq_inter.name("intermediate matrix general rational");
	_mgr_inter.name("intermediate matrix general real");
	_mcz_inter.name("intermediate matrix complex integer");
	_mcq_inter.name("intermediate matrix complex rational");
	_mcr_inter.name("intermediate matrix complex real");
	_mcgq_inter.name("intermediate matrix complex general rational");
	_mcgr_inter.name("intermediate matrix complex general real");

#ifdef	ZHP_DEBUG

	debug(_start);

	debug(_node_expr);
	debug(_node_term);
	debug(_node_opd);

	debug(_plus);
	debug(_minus);
	debug(_times);
	debug(_divide);
	debug(_power);

	debug(_o_str);

	debug(_o_z);
	debug(_o_q);
	debug(_o_r);
	debug(_o_cz);
	debug(_o_cq);
	debug(_o_cr);

	debug(_o_vz);
	debug(_o_vq);
	debug(_o_vr);
	debug(_o_vgq);
	debug(_o_vgr);
	debug(_o_vcz);
	debug(_o_vcq);
	debug(_o_vcr);
	debug(_o_vcgq);
	debug(_o_vcgr);

	debug(_o_mz);
	debug(_o_mq);
	debug(_o_mr);
	debug(_o_mgq);
	debug(_o_mgr);
	debug(_o_mz);
	debug(_o_mcq);
	debug(_o_mcr);
	debug(_o_mcgq);
	debug(_o_mcgr);

	debug(_str);
	
	debug(_z);
	debug(_q);
	debug(_r);
	debug(_gq);
	debug(_gr);
	debug(_cz);
	debug(_cq);
	debug(_cr);
	debug(_cgq);
	debug(_cgr);

	debug(_vz);
	debug(_vq);
	debug(_vr);
	debug(_vgq);
	debug(_vgr);
	debug(_vcz);
	debug(_vcq);
	debug(_vcr);
	debug(_vcgq);
	debug(_vcgr);

	debug(_mz);
	debug(_mq);
	debug(_mr);
	debug(_mgq);
	debug(_mgr);
	debug(_mz);
	debug(_mcq);
	debug(_mcr);
	debug(_mcgq);
	debug(_mcgr);

#endif

}

}
