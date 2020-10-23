#ifndef CONFIG_H_
#define CONFIG_H_

// C++ Standard Libraries
#include <cmath>
#include <functional>
#include <map>
#include <string>

// Engine Headers
#include "operand.hpp"
#include "operation.hpp"
#include "variable.hpp"
// #include "Function.hpp"

enum opcode {
	op_nao,
	op_add,
	op_sub,
	op_mul,
	op_div,
	op_exp,
	op_sin,
	op_cos,
	op_tan,
	op_csc,
	op_sec,
	op_cot,
	op_log,
	op_sum,
	op_fac
};

enum tcode {
	t_nat,
	t_opd,
	t_opn,
	t_var,
	t_ftr
};

// Forward Declarations
template <class T, class U>
class Function;

/**
 * @brief A class used as a method
 * of storing the common set of
 * operations and real/complex space
 * constants.
 */
template <class T>
class config {
public:
	using opd = Operand <T>;
	using opn = operation;
	using var = Variable <T>;
	using ftr = Function <T, int>;

	struct specs {
		std::string in;
		std::string out;
		std::size_t ops;
		std::function <T (const std::vector <Token *> &)> opn;

		opcode ocode;
	};
	
	T negative;
	T zero;
	T one;
private:
	std::map <int, specs> tbl;
public:
	config();
	config(const std::vector <specs> &, const T &,
			const T &, const T &);

	opcode code(const std::string &) const;

	opn *alloc_opn(opcode) const;
	opn *alloc_opn(const std::string &) const;
	
	static tcode code(Token *);

	static T read(const std::string &);

	// auto convertion
	// static tuple <T
};

using namespace std;

template <class T>
config <T> ::config() : config({
		{"+", "$1 + $2", 2, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd && code(ins[1]) != t_opd)
				throw typename opn::Token_mismatch();
			return (dynamic_cast <opd *> (ins[0]))->get()
				+ (dynamic_cast <opd *> (ins[1]))->get();
		}, op_add},
		{"-", "$1 - $2", 2, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd && code(ins[1]) != t_opd)
				throw typename opn::Token_mismatch();
			return (dynamic_cast <opd *> (ins[0]))->get()
				- (dynamic_cast <opd *> (ins[1]))->get();
		}, op_sub},
		{"*", "$1 * $2", 2, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd && code(ins[1]) != t_opd)
				throw typename opn::Token_mismatch();
			return (dynamic_cast <opd *> (ins[0]))->get()
				* (dynamic_cast <opd *> (ins[1]))->get();
		}, op_mul},
		{"/", "$1 / $2", 2, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd && code(ins[1]) != t_opd)
				throw typename opn::Token_mismatch();
			return (dynamic_cast <opd *> (ins[0]))->get()
				/ (dynamic_cast <opd *> (ins[1]))->get();
		}, op_div},
		{"^", "$1 ^ $2", 2, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd && code(ins[1]) != t_opd)
				throw typename opn::Token_mismatch();
			return pow((dynamic_cast <opd *> (ins[0]))->get(),
				(dynamic_cast <opd *> (ins[1]))->get());
		}, op_exp},
		{"sin", "sin $1", 1, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd)
				throw typename opn::Token_mismatch();
			return sin((dynamic_cast <opd *> (ins[0]))->get());
		}, op_sin},
		{"cos", "cos $1", 1, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd)
				throw typename opn::Token_mismatch();
			return cos((dynamic_cast <opd *> (ins[0]))->get());
		}, op_cos},
		{"tan", "tan $1", 1, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd)
				throw typename opn::Token_mismatch();
			return tan((dynamic_cast <opd *> (ins[0]))->get());
		}, op_tan},
		{"csc", "csc $1", 1, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd)
				throw typename opn::Token_mismatch();
			return T(1) / sin((dynamic_cast <opd *> (ins[0]))->get());
		}, op_csc},
		{"sec", "sec $1", 1, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd)
				throw typename opn::Token_mismatch();
			return T(1) / cos((dynamic_cast <opd *> (ins[0]))->get());
		}, op_sec},
		{"cot", "cot $1", 1, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd)
				throw typename opn::Token_mismatch();
			return T(1) / tan((dynamic_cast <opd *> (ins[0]))->get());
		}, op_cot},
		{"log", "log_$1 $2", 2, [](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd && code(ins[1]) != t_opd)
				throw typename opn::Token_mismatch();
			return log((dynamic_cast <opd *> (ins[1]))->get())
				/ log((dynamic_cast <opd *> (ins[0]))->get());
		}, op_log},
		{"sum", "sum^{$3}_{$1 = $2} $4", 4, [&](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_var || code(ins[1]) != t_opd ||
				code(ins[2]) != t_opd || code(ins[3]) != t_ftr)
				throw typename opn::Token_mismatch();

			opd *start = dynamic_cast <opd *> (ins[1]);
			opd *end = dynamic_cast <opd *> (ins[2]);
			
			// ftr *expr = dynamic_cast <ftr *> (ins[3]);

			T value = zero;
			
			/* for (T i = start->get(); i <= end->get(); i += T(1))
				value += (*expr)(i); */
			
			return value;
		}, op_sum},
		{"!", "$1!", 1, [&](const std::vector <Token *> &ins) {
			if (code(ins[0]) != t_opd)
				throw typename opn::Token_mismatch();

			opd *start = dynamic_cast <opd *> (ins[0]);

			size_t value = 1;
			for (size_t i = (double) start->get(); i > 0; i--)
				value *= i;

			return T(value);
		}, op_fac}}, T(-1), T(0), T(1)) {}

template <class T>
config <T> ::config(const std::vector <specs> &st, const T &n,
		const T &z, const T &o) : negative(n),
		zero(z), one(o)	
{
	std::size_t counter = 0;
	for (const auto &spec : st)
		tbl[counter++] = spec;
}

template <class T>
opcode config <T> ::code(const std::string &str) const
{
	auto itr = std::find_if(tbl.begin(), tbl.end(),
		[&](const std::pair <int, specs> &pr) {
			return pr.second.in == str;
		}
	);

	if (itr == tbl.end())
		return op_nao;
	return itr->second.ocode;
}

template <class T>
typename config <T> ::opn *config <T> ::alloc_opn(opcode ocode) const
{
	auto itr = std::find_if(tbl.begin(), tbl.end(),
		[&](const std::pair <int, specs> &pr) {
			return pr.second.ocode == ocode;
		}
	);

	if (itr == tbl.end())
		return nullptr;

	specs sp = itr->second;

	return new opn(sp.in, sp.out, sp.ops, sp.opn);
}

template <class T>
typename config <T> ::opn *config <T> ::alloc_opn(const std::string &str) const
{
	auto itr = std::find_if(tbl.begin(), tbl.end(),
		[&](const std::pair <int, specs> &pr) {
			return pr.second.in == str;
		}
	);

	if (itr == tbl.end())
		return nullptr;

	specs sp = itr->second;

	return new opn(sp.in, sp.out, sp.ops, sp.opn);
}

template <class T>
tcode config <T> ::code(Token *t)
{
	if (dynamic_cast <opd *> (t))
		return t_opd;
	if (dynamic_cast <opn *> (t))
		return t_opn;
	if (dynamic_cast <var *> (t))
		return t_var;
	// if (dynamic_cast <ftr *> (t))
	// 	return t_ftr;
	return t_nat;
}

#endif
