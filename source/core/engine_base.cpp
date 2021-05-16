#include <core/engine_base.hpp>

namespace zhetapi {

// TODO: put this constructor in another file
engine_base::engine_base()
{
	std::vector <std::pair <std::pair <std::string, std::vector <std::type_index>>, Token *>> ops;

	//////////////////////////////////////////
	// Real Scalar Arithemtic
	//////////////////////////////////////////

	// TODO: Really shorten the maCmpRos
	
	// Addition and Subtraction
	_add_binary_operation_set(+);
	_add_binary_operation_set(-);

	// InCmpRement and deCmpRement
	_add_unary_operation_ftr(p++, Z, Z, in->get() + 1);
	_add_unary_operation_ftr(p--, Z, Z, in->get() - 1);
	
	_add_unary_operation_ftr(r++, Z, Z, in->get() + 1);
	_add_unary_operation_ftr(r--, Z, Z, in->get() - 1);

	// Modular arithmetic
	_add_binary_operation(%, Z, Z, Z);

	// Multiplication
	_add_binary_operation(*, Z, Z, Z);
	_add_binary_operation(*, R, R, R);
	_add_binary_operation(*, Q, Q, Q);
	_add_binary_operation(*, CmpR, CmpR, CmpR);
	_add_binary_operation(*, CmpQ, CmpQ, CmpQ);

	_add_heterogenous_binary_operation(*, R, Z, R);
	_add_heterogenous_binary_operation(*, Z, Q, Q);
	_add_heterogenous_binary_operation_ftr(*, R, Q, R, a->get() * (R)
			b->get());
	_add_heterogenous_binary_operation(*, R, CmpR, CmpR);
	_add_heterogenous_binary_operation_ftr(*, R, CmpQ, CmpR, CmpR(a->get() * (R)
				b->get().real(), a->get() * (R)
				b->get().imag()));
	_add_heterogenous_binary_operation_ftr(*, Z, CmpR, CmpR, CmpR(a->get() * (R)
				b->get().real(), a->get() * (R)
				b->get().imag()));
	_add_heterogenous_binary_operation_ftr(*, Z, CmpQ, CmpQ, CmpQ((Q) a->get() *
				b->get().real(), (Q) a->get() *
				b->get().imag()));
	_add_heterogenous_binary_operation_ftr(*, Q, CmpR, CmpR, CmpR((R) a->get() *
				b->get().real(), (R) a->get() *
				b->get().imag()));
	_add_heterogenous_binary_operation(*, Q, CmpQ, CmpQ);

	// Division
	_add_binary_operation(/, R, R, R);
	_add_binary_operation(/, Q, Q, Q);
	_add_binary_operation(/, CmpR, CmpR, CmpR);
	_add_binary_operation(/, CmpQ, CmpQ, CmpQ);
	
	_add_binary_operation_ftr(/, Z, Z, Q, ((Q) a->get()) / ((Q) b->get()));

	_add_heterogenous_binary_operation(/, R, Z, R);
	_add_heterogenous_binary_operation_ftr(/, Z, Q, Q, ((Q)
				a->get()) / ((Q) b->get()));
	_add_heterogenous_binary_operation_ftr(/, R, Q, R, a->get() * (R)
			b->get());
	_add_heterogenous_binary_operation(/, R, CmpR, CmpR);
	_add_heterogenous_binary_operation_ftr(/, R, CmpQ, CmpR, CmpR(a->get() * (R)
				b->get().real(), a->get() * (R)
				b->get().imag()));
	_add_heterogenous_binary_operation_ftr(/, Z, CmpR, CmpR, CmpR(a->get() * (R)
				b->get().real(), a->get() * (R)
				b->get().imag()));
	_add_heterogenous_binary_operation_ftr(/, Z, CmpQ, CmpQ, CmpQ((Q) a->get() *
				b->get().real(), (Q) a->get() *
				b->get().imag()));
	_add_heterogenous_binary_operation_ftr(/, Q, CmpR, CmpR, CmpR((R) a->get() *
				b->get().real(), (R) a->get() *
				b->get().imag()));
	_add_heterogenous_binary_operation(/, Q, CmpQ, CmpQ);

	// Exponentiation
	_add_binary_operation_ftr(^, Z, Z, Z, (Z) pow(a->get(), b->get()));
	_add_binary_operation_ftr(^, R, R, R, pow(a->get(), b->get()));
	_add_binary_operation_ftr(^, R, Z, R, pow(a->get(), b->get()));
	_add_binary_operation_ftr(^, Q, Q, R, pow((R) a->get(), (R) b->get()));

	// FIXME: Wrong values
	_add_binary_operation_ftr(^, CmpR, CmpR, CmpR, 1);
	_add_binary_operation_ftr(^, CmpQ, CmpQ, CmpQ, Q(1));
	
	// TODO: Preserve rationality for this overload
	_add_heterogenous_binary_operation_ftr(^, Q, Z, R, pow((R) a->get(), (R) b->get()));

	// Trigonometry
	_add_unary_operation_ftr(sin, Z, R, std::sin(R (in->get())));
	_add_unary_operation_ftr(sin, Q, R, std::sin(R (in->get())));
	_add_unary_operation_ftr(sin, R, R, std::sin(R (in->get())));
	
	_add_unary_operation_ftr(cos, Z, R, std::cos(R (in->get())));
	_add_unary_operation_ftr(cos, Q, R, std::cos(R (in->get())));
	_add_unary_operation_ftr(cos, R, R, std::cos(R (in->get())));
	
	_add_unary_operation_ftr(tan, Z, R, std::tan(R (in->get())));
	_add_unary_operation_ftr(tan, Q, R, std::tan(R (in->get())));
	_add_unary_operation_ftr(tan, R, R, std::tan(R (in->get())));
	
	_add_unary_operation_ftr(csc, Z, R, 1/std::sin(in->get()));
	_add_unary_operation_ftr(csc, Q, R, 1/std::sin(R (in->get())));
	_add_unary_operation_ftr(csc, R, R, 1/std::sin(in->get()));
	
	_add_unary_operation_ftr(sec, Z, R, 1/std::cos(in->get()));
	_add_unary_operation_ftr(sec, Q, R, 1/std::cos(R (in->get())));
	_add_unary_operation_ftr(sec, R, R, 1/std::cos(in->get()));
	
	_add_unary_operation_ftr(cot, Z, R, 1/std::tan(in->get()));
	_add_unary_operation_ftr(cot, Q, R, 1/std::tan(R (in->get())));
	_add_unary_operation_ftr(cot, R, R, 1/std::tan(in->get()));

	// Hyperbolic Functions
	_add_unary_operation(sinh, Z, R);
	_add_unary_operation_ftr(sinh, Q, R, sinh(R (in->get())));
	_add_unary_operation(sinh, R, R);
	
	_add_unary_operation(cosh, Z, R);
	_add_unary_operation_ftr(cosh, Q, R, cosh(R (in->get())));
	_add_unary_operation(cosh, R, R);
	
	_add_unary_operation(tanh, Z, R);
	_add_unary_operation_ftr(tanh, Q, R, tanh(R (in->get())));
	_add_unary_operation(tanh, R, R);
	
	_add_unary_operation_ftr(csch, Z, R, 1/sinh(in->get()));
	_add_unary_operation_ftr(csch, Q, R, 1/sinh(R (in->get())));
	_add_unary_operation_ftr(csch, R, R, 1/sinh(in->get()));
	
	_add_unary_operation_ftr(sech, Z, R, 1/cosh(in->get()));
	_add_unary_operation_ftr(sech, Q, R, 1/cosh(R (in->get())));
	_add_unary_operation_ftr(sech, R, R, 1/cosh(in->get()));
	
	_add_unary_operation_ftr(coth, Z, R, 1/tanh(in->get()));
	_add_unary_operation_ftr(coth, Q, R, 1/tanh(R (in->get())));
	_add_unary_operation_ftr(coth, R, R, 1/tanh(in->get()));

	// Logarithms
	_add_unary_operation_ftr(ln, Z, R, log(R (in->get())));
	_add_unary_operation_ftr(ln, Q, R, log(R (in->get())));
	_add_unary_operation_ftr(ln, R, R, log(R (in->get())));
	
	_add_unary_operation_ftr(log, Z, R, log10(R (in->get())));
	_add_unary_operation_ftr(log, Q, R, log10(R (in->get())));
	_add_unary_operation_ftr(log, R, R, log10(R (in->get())));
	
	_add_binary_operation_ftr(log, Z, Z, R, log(R (b->get()))/log(R (a->get())));
	_add_binary_operation_ftr(log, Q, Q, R, log(R (b->get()))/log(R (a->get())));
	_add_binary_operation_ftr(log, R, R, R, log(R (b->get()))/log(R (a->get())));

	_add_heterogenous_binary_operation_ftr(log, Z, R, R, log(R (b->get()))/log(R (a->get())));
	_add_heterogenous_binary_operation_ftr(log, Z, Q, R, log(R (b->get()))/log(R (a->get())));
	_add_heterogenous_binary_operation_ftr(log, Q, R, R, log(R (b->get()))/log(R (a->get())));
	
	_add_unary_operation_ftr(lg, Z, R, log(R (in->get()))/log(2));
	_add_unary_operation_ftr(lg, Q, R, log(R (in->get()))/log(2));
	_add_unary_operation_ftr(lg, R, R, log(R (in->get()))/log(2));

	// Misc (TODO: abs for Q should return Q)
	_add_unary_operation_ftr(abs, Z, Z, std::abs(in->get()));
	_add_unary_operation_ftr(abs, Q, R, std::abs(in->get()));
	_add_unary_operation_ftr(abs, R, R, std::abs(in->get()));
	
	// TODO: check for perfect squares in Z and Q
	// TODO: add std::sqrt for Q
	// TODO: remove output type restrictions
	_add_unary_operation_ftr(sqrt, Z, R, std::sqrt(in->get()));
	_add_unary_operation_ftr(sqrt, Q, R, std::sqrt((R) in->get()));
	_add_unary_operation_ftr(sqrt, R, R, std::sqrt(in->get()));
	
	_add_unary_operation_ftr(round, Z, Z, std::round(in->get()));
	_add_unary_operation_ftr(round, Q, Z, std::round((R) in->get()));
	_add_unary_operation_ftr(round, R, Z, std::round(in->get()));

	_add_unary_operation_ftr(floor, Z, Z, std::floor(in->get()));
	_add_unary_operation_ftr(floor, Q, Z, std::floor((R) in->get()));
	_add_unary_operation_ftr(floor, R, Z, std::floor(in->get()));

	_add_unary_operation_ftr(ceil, Z, Z, std::ceil(in->get()));
	_add_unary_operation_ftr(ceil, Q, Z, std::ceil((R) in->get()));
	_add_unary_operation_ftr(ceil, R, Z, std::ceil(in->get()));
	
	//////////////////////////////////////////
	// Vector operations
	//////////////////////////////////////////

	// Addition
	_add_binary_operation(+, VecZ, VecZ, VecZ);
	_add_binary_operation(+, VecQ, VecQ, VecQ);
	_add_binary_operation(+, VecR, VecR, VecR);
	
	// Subtraction
	_add_binary_operation(-, VecZ, VecZ, VecZ);
	_add_binary_operation(-, VecQ, VecQ, VecQ);
	_add_binary_operation(-, VecR, VecR, VecR);

	// Scalar multiplication
	_add_binary_operation_ftr(*, Z, VecZ, VecZ, (a->get()) * b->get());
	_add_binary_operation_ftr(*, Z, VecQ, VecQ, ((Q) a->get()) * b->get());
	
	// Dot product
	_add_binary_operation_ftr(., VecZ, VecZ, Z, inner(a->get(), b->get()));
	_add_binary_operation_ftr(., VecQ, VecQ, Q, inner(a->get(), b->get()));
	_add_binary_operation_ftr(., VecR, VecR, R, inner(a->get(), b->get()));
	
	// Shur product
	_add_binary_operation_ftr(shur, VecQ, VecQ, VecQ, shur(a->get(), b->get()));
	_add_binary_operation_ftr(shur, VecR, VecR, VecR, shur(a->get(), b->get()));

	// Transposition
	_add_unary_operation_ftr(transpose, VecQ, MatQ, in->get().transpose());
	_add_unary_operation_ftr(transpose, VecR, MatR, in->get().transpose());
	
	//////////////////////////////////////////
	// Matrix operations
	//////////////////////////////////////////

	_add_binary_operation(+, MatQ, MatQ, MatQ);
	_add_binary_operation(+, MatR, MatR, MatR);
	
	_add_binary_operation(-, MatQ, MatQ, MatQ);
	_add_binary_operation(-, MatR, MatR, MatR);
	
	_add_binary_operation_ftr(shur, MatQ, MatQ, MatQ, shur(a->get(), b->get()));
	_add_binary_operation_ftr(shur, MatR, MatR, MatR, shur(a->get(), b->get()));

	_add_unary_operation_ftr(transpose, MatQ, MatQ, in->get().transpose());
	_add_unary_operation_ftr(transpose, MatR, MatR, in->get().transpose());

	// Other Linear Algebra
	_add_binary_operation(*, MatQ, VecQ, MatQ);
	_add_binary_operation(*, VecQ, MatQ, MatQ);

	//////////////////////////////////////////
	// Combinatorial Operations
	//////////////////////////////////////////

	_add_unary_operation_ftr(!, Z, Z, utility::integral_factorial(in->get())); 
	
	_add_binary_operation_ftr(binom, Z, Z, Z, utility::integral_binom(a->get(), b->get()));

	//////////////////////////////////////////
	// Boolean operations
	//////////////////////////////////////////

	// Add a "variant" type

	// == and != operators
	_universal["=="] = new operation(
		"==",
		"$1 == $2",
		2,
		[](const std::vector <Token *> &ins) {
			return new Operand <bool> (tokcmp(ins[0], ins[1]));
		}
	);

	_universal["!="] = new operation(
		"!=",
		"$1 != $2",
		2,
		[](const std::vector <Token *> &ins) {
			return new Operand <bool> (!tokcmp(ins[0], ins[1]));
		}
	);

	// Make these universal operations
	_add_binary_operation(>, Z, Z, B);
	_add_binary_operation(>, R, R, B);
	_add_binary_operation(>, Q, Q, B);
	
	_add_binary_operation(<, Z, Z, B);
	_add_binary_operation(<, R, R, B);
	_add_binary_operation(<, Q, Q, B);

	_add_binary_operation(>=, Z, Z, B);
	_add_binary_operation(<=, Z, Z, B);
	
	_add_heterogenous_binary_operation(>, R, Z, B);
	_add_heterogenous_binary_operation(<, R, Z, B);
	_add_heterogenous_binary_operation(>, Q, Z, B);
	_add_heterogenous_binary_operation(<, Q, Z, B);
	_add_heterogenous_binary_operation(>, R, Q, B);
	_add_heterogenous_binary_operation(<, R, Q, B);

	// Boolean operations
	_add_binary_operation(||, B, B, B);
	_add_binary_operation(&&, B, B, B);

	//////////////////////////////////////////
	// Transfer into table
	//////////////////////////////////////////

	for (auto pr : ops) {
		operation *opn = dynamic_cast <operation *> (pr.second);

		if (_overloads.count(pr.first.first))
			_overloads[pr.first.first].push_back({pr.first.second, pr.second});
		else
			_overloads.insert(std::pair <std::string, std::vector <std::pair <signature, Token *>>> {pr.first.first, std::vector <std::pair <signature, Token *>> {{pr.first.second, pr.second}}});
	}
}

engine_base::engine_base(const engine_base &other)
		: _overloads(other._overloads) {}

engine_base &engine_base::operator=(const engine_base &other)
{
	if (this != &other)
		_overloads = other._overloads;

	return *this;
}

engine_base::~engine_base()
{
	for (auto overload_list : _overloads) {
		for (auto id : overload_list.second)
			delete id.second;
	}
}

Token *engine_base::compute(
		const std::string &str,
		const std::vector <Token *> &args)
{
	// The actual operation
	operation *optr = nullptr;

	// Check presence in universal operation sets
	if (_universal.find(str) != _universal.end()) {
		optr = dynamic_cast <operation *> (_universal[str]);

		return optr->compute(args);
	}
	
	// Generature the signature
	signature sig = gen_signature(args);

	// Take address to avoid copy
	overloads *ovlds = &_overloads[str];

	size_t len = sig.size();
	for (auto itr = ovlds->begin(); itr != ovlds->end(); itr++) {
		if (itr->first.size() != len)
			continue;

		bool ps = true;

		for (size_t i = 0; i < len; i++) {					
			if (sig[i] != itr->first[i]) {
				ps = false;

				break;
			}
		}

		if (ps) {
			optr = dynamic_cast <operation *> (itr->second);
			
			break;
		}
	}
	
	if (optr)
		return optr->compute(args);
	
	throw unknown_op_overload(gen_overload_msg(sig, str));
}

bool engine_base::present(const std::string &str) const
{
	return _overloads.find(str) != _overloads.end();
}

std::string engine_base::overload_catalog(const std::string &str)
{
	std::string out = "Available overloads for \"" + str + "\": {";

	overloads ovlds = _overloads[str];
	for (size_t i = 0; i < ovlds.size(); i++) {
		signature sig = ovlds[i].first;

		out += "(";

		for (size_t j = 0; j < sig.size(); j++) {
			out += types::symbol(sig[j]);

			if (j < sig.size() - 1)
				out += ",";
		}

		out += ")";

		if (i < ovlds.size() - 1)
			out += ", ";
	}

	return out + "}";
}

// Private methods
std::string engine_base::gen_overload_msg(const signature &sig, const std::string &str)
{
	std::string msg = "Unknown overload (";

	for (size_t i = 0; i < sig.size(); i++) {
		msg += types::symbol(sig[i]);
		
		if (i < sig.size() - 1)
			msg += ", ";
	}

	return msg + ") for operation \"" + str + "\". " + overload_catalog(str);
}

signature engine_base::gen_signature(const std::vector <Token *> &vals)
{
	signature sig;

	for (Token *tptr : vals) {
		if (!tptr)
			throw gensig_nullptr();
		sig.push_back(typeid(*tptr));
	}

	return sig;
}

}
