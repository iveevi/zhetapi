#include <barn.hpp>
#include <core/algorithm.hpp>
#include <function.hpp>

#include <cmath>

namespace zhetapi {

//////////////////////////////////////////
// Constructors
//////////////////////////////////////////

Barn::Barn()
{
	//////////////////////////////////////////
	// Real Scalar Arithemtic
	//////////////////////////////////////////
	
	// Addition and Subtraction
	__add_binary_operation_set(+);
	__add_binary_operation_set(-);

	// Increment and decrement
	__add_unary_operation_ftr(p++, Z, Z, in->get() + 1);
	__add_unary_operation_ftr(p--, Z, Z, in->get() - 1);
	
	__add_unary_operation_ftr(r++, Z, Z, in->get() + 1);
	__add_unary_operation_ftr(r--, Z, Z, in->get() - 1);

	// Modular arithmetic
	__add_binary_operation(%, Z, Z, Z);

	// Multiplication
	__add_binary_operation(*, Z, Z, Z);
	__add_binary_operation(*, R, R, R);
	__add_binary_operation(*, Q, Q, Q);
	__add_binary_operation(*, CR, CR, CR);
	__add_binary_operation(*, CQ, CQ, CQ);

	__add_heterogenous_binary_operation(*, R, Z, R);
	__add_heterogenous_binary_operation(*, Z, Q, Q);
	__add_heterogenous_binary_operation_ftr(*, R, Q, R, a->get() * (R)
			b->get());
	__add_heterogenous_binary_operation(*, R, CR, CR);
	__add_heterogenous_binary_operation_ftr(*, R, CQ, CR, CR(a->get() * (R)
				b->get().real(), a->get() * (R)
				b->get().imag()));
	__add_heterogenous_binary_operation_ftr(*, Z, CR, CR, CR(a->get() * (R)
				b->get().real(), a->get() * (R)
				b->get().imag()));
	__add_heterogenous_binary_operation_ftr(*, Z, CQ, CQ, CQ((Q) a->get() *
				b->get().real(), (Q) a->get() *
				b->get().imag()));
	__add_heterogenous_binary_operation_ftr(*, Q, CR, CR, CR((R) a->get() *
				b->get().real(), (R) a->get() *
				b->get().imag()));
	__add_heterogenous_binary_operation(*, Q, CQ, CQ);

	// Division
	__add_binary_operation(/, R, R, R);
	__add_binary_operation(/, Q, Q, Q);
	__add_binary_operation(/, CR, CR, CR);
	__add_binary_operation(/, CQ, CQ, CQ);
	
	__add_binary_operation_ftr(/, Z, Z, Q, ((Q) a->get()) / ((Q) b->get()));

	__add_heterogenous_binary_operation(/, R, Z, R);
	__add_heterogenous_binary_operation_ftr(/, Z, Q, Q, ((Q)
				a->get()) / ((Q) b->get()));
	__add_heterogenous_binary_operation_ftr(/, R, Q, R, a->get() * (R)
			b->get());
	__add_heterogenous_binary_operation(/, R, CR, CR);
	__add_heterogenous_binary_operation_ftr(/, R, CQ, CR, CR(a->get() * (R)
				b->get().real(), a->get() * (R)
				b->get().imag()));
	__add_heterogenous_binary_operation_ftr(/, Z, CR, CR, CR(a->get() * (R)
				b->get().real(), a->get() * (R)
				b->get().imag()));
	__add_heterogenous_binary_operation_ftr(/, Z, CQ, CQ, CQ((Q) a->get() *
				b->get().real(), (Q) a->get() *
				b->get().imag()));
	__add_heterogenous_binary_operation_ftr(/, Q, CR, CR, CR((R) a->get() *
				b->get().real(), (R) a->get() *
				b->get().imag()));
	__add_heterogenous_binary_operation(/, Q, CQ, CQ);

	// Exponentiation
	__add_binary_operation_ftr(^, Z, Z, Z, (Z) pow(a->get(), b->get()));
	__add_binary_operation_ftr(^, R, R, R, pow(a->get(), b->get()));
	__add_binary_operation_ftr(^, R, Z, R, pow(a->get(), b->get()));
	__add_binary_operation_ftr(^, Q, Q, R, pow((R) a->get(), (R) b->get()));
	__add_binary_operation_ftr(^, CR, CR, CR, 1);
	__add_binary_operation_ftr(^, CQ, CQ, CQ, Q(1));

	// Trigonometry
	__add_unary_operation_ftr(sin, Z, R, std::sin(R (in->get())));
	__add_unary_operation_ftr(sin, Q, R, std::sin(R (in->get())));
	__add_unary_operation_ftr(sin, R, R, std::sin(R (in->get())));
	
	__add_unary_operation_ftr(cos, Z, R, std::cos(R (in->get())));
	__add_unary_operation_ftr(cos, Q, R, std::cos(R (in->get())));
	__add_unary_operation_ftr(cos, R, R, std::cos(R (in->get())));
	
	__add_unary_operation_ftr(tan, Z, R, std::tan(R (in->get())));
	__add_unary_operation_ftr(tan, Q, R, std::tan(R (in->get())));
	__add_unary_operation_ftr(tan, R, R, std::tan(R (in->get())));
	
	__add_unary_operation_ftr(csc, Z, R, 1/std::sin(in->get()));
	__add_unary_operation_ftr(csc, Q, R, 1/std::sin(R (in->get())));
	__add_unary_operation_ftr(csc, R, R, 1/std::sin(in->get()));
	
	__add_unary_operation_ftr(sec, Z, R, 1/std::cos(in->get()));
	__add_unary_operation_ftr(sec, Q, R, 1/std::cos(R (in->get())));
	__add_unary_operation_ftr(sec, R, R, 1/std::cos(in->get()));
	
	__add_unary_operation_ftr(cot, Z, R, 1/std::tan(in->get()));
	__add_unary_operation_ftr(cot, Q, R, 1/std::tan(R (in->get())));
	__add_unary_operation_ftr(cot, R, R, 1/std::tan(in->get()));

	// Hyperbolic Functions
	__add_unary_operation(sinh, Z, R);
	__add_unary_operation_ftr(sinh, Q, R, sinh(R (in->get())));
	__add_unary_operation(sinh, R, R);
	
	__add_unary_operation(cosh, Z, R);
	__add_unary_operation_ftr(cosh, Q, R, cosh(R (in->get())));
	__add_unary_operation(cosh, R, R);
	
	__add_unary_operation(tanh, Z, R);
	__add_unary_operation_ftr(tanh, Q, R, tanh(R (in->get())));
	__add_unary_operation(tanh, R, R);
	
	__add_unary_operation_ftr(csch, Z, R, 1/sinh(in->get()));
	__add_unary_operation_ftr(csch, Q, R, 1/sinh(R (in->get())));
	__add_unary_operation_ftr(csch, R, R, 1/sinh(in->get()));
	
	__add_unary_operation_ftr(sech, Z, R, 1/cosh(in->get()));
	__add_unary_operation_ftr(sech, Q, R, 1/cosh(R (in->get())));
	__add_unary_operation_ftr(sech, R, R, 1/cosh(in->get()));
	
	__add_unary_operation_ftr(coth, Z, R, 1/tanh(in->get()));
	__add_unary_operation_ftr(coth, Q, R, 1/tanh(R (in->get())));
	__add_unary_operation_ftr(coth, R, R, 1/tanh(in->get()));

	// Logarithms
	__add_unary_operation_ftr(ln, Z, R, log(R (in->get())));
	__add_unary_operation_ftr(ln, Q, R, log(R (in->get())));
	__add_unary_operation_ftr(ln, R, R, log(R (in->get())));
	
	__add_unary_operation_ftr(log, Z, R, log10(R (in->get())));
	__add_unary_operation_ftr(log, Q, R, log10(R (in->get())));
	__add_unary_operation_ftr(log, R, R, log10(R (in->get())));
	
	__add_binary_operation_ftr(log, Z, Z, R, log(R (b->get()))/log(R (a->get())));
	__add_binary_operation_ftr(log, Q, Q, R, log(R (b->get()))/log(R (a->get())));
	__add_binary_operation_ftr(log, R, R, R, log(R (b->get()))/log(R (a->get())));

	__add_heterogenous_binary_operation_ftr(log, Z, R, R, log(R (b->get()))/log(R (a->get())));
	__add_heterogenous_binary_operation_ftr(log, Z, Q, R, log(R (b->get()))/log(R (a->get())));
	__add_heterogenous_binary_operation_ftr(log, Q, R, R, log(R (b->get()))/log(R (a->get())));
	
	__add_unary_operation_ftr(lg, Z, R, log(R (in->get()))/log(2));
	__add_unary_operation_ftr(lg, Q, R, log(R (in->get()))/log(2));
	__add_unary_operation_ftr(lg, R, R, log(R (in->get()))/log(2));
	
	//////////////////////////////////////////
	// Vector operations
	//////////////////////////////////////////

	// Addition
	__add_binary_operation(+, VZ, VZ, VZ);
	__add_binary_operation(+, VQ, VQ, VQ);
	__add_binary_operation(+, VR, VR, VR);
	
	// Subtraction
	__add_binary_operation(-, VZ, VZ, VZ);
	__add_binary_operation(-, VQ, VQ, VQ);
	__add_binary_operation(-, VR, VR, VR);

	// Scalar multiplication
	__add_binary_operation_ftr(*, Z, VZ, VZ, (a->get()) * b->get());
	__add_binary_operation_ftr(*, Z, VQ, VQ, ((Q) a->get()) * b->get());
	
	// Dot product
	__add_binary_operation_ftr(., VZ, VZ, Z, inner(a->get(), b->get()));
	__add_binary_operation_ftr(., VQ, VQ, Q, inner(a->get(), b->get()));
	__add_binary_operation_ftr(., VR, VR, R, inner(a->get(), b->get()));
	
	// Shur product
	__add_binary_operation_ftr(shur, VQ, VQ, VQ, shur(a->get(), b->get()));
	__add_binary_operation_ftr(shur, VR, VR, VR, shur(a->get(), b->get()));

	// Transposition
	__add_unary_operation_ftr(transpose, VQ, MQ, in->get().transpose());
	__add_unary_operation_ftr(transpose, VR, MR, in->get().transpose());
	
	//////////////////////////////////////////
	// Matrix operations
	//////////////////////////////////////////

	__add_binary_operation(+, MQ, MQ, MQ);
	__add_binary_operation(+, MR, MR, MR);
	
	__add_binary_operation(-, MQ, MQ, MQ);
	__add_binary_operation(-, MR, MR, MR);
	
	__add_binary_operation_ftr(shur, MQ, MQ, MQ, shur(a->get(), b->get()));
	__add_binary_operation_ftr(shur, MR, MR, MR, shur(a->get(), b->get()));

	__add_unary_operation_ftr(transpose, MQ, MQ, in->get().transpose());
	__add_unary_operation_ftr(transpose, MR, MR, in->get().transpose());

	// Other Linear Algebra
	__add_binary_operation(*, MQ, VQ, MQ);
	__add_binary_operation(*, VQ, MQ, MQ);

	//////////////////////////////////////////
	// Combinatorial Operations
	//////////////////////////////////////////

	__add_unary_operation_ftr(!, Z, Z, utility::integral_factorial(in->get())); 
	
	__add_binary_operation_ftr(binom, Z, Z, Z, utility::integral_binom(a->get(), b->get()));

	//////////////////////////////////////////
	// Boolean operations
	//////////////////////////////////////////

	// Add a "variant" type
	__add_binary_operation(==, Z, Z, B);
	__add_binary_operation(!=, Z, Z, B);
	__add_binary_operation(>, Z, Z, B);
	__add_binary_operation(>=, Z, Z, B);
	__add_binary_operation(<, Z, Z, B);
	__add_binary_operation(<=, Z, Z, B);

	//////////////////////////////////////////
	// Transfer into table
	//////////////////////////////////////////

	for (auto pr : ops) {
		operation *opn = dynamic_cast <operation *> (pr.second);

		if (table.count(pr.first.first))
			table[pr.first.first].push_back({pr.first.second, pr.second});
		else
			table.insert(::std::pair <::std::string, ::std::vector <::std::pair <signature, Token *>>> {pr.first.first, ::std::vector <::std::pair <signature, Token *>> {{pr.first.second, pr.second}}});
	}
}

Barn::Barn(const Barn &other) :
		__reg_table(other.__reg_table),
		__alg_table(other.__alg_table),
		__ftr_table(other.__ftr_table),
		__var_table(other.__var_table)
{
	for (auto pr : other.ops)
		ops.push_back({pr.first, pr.second->copy()});
	
	for (auto pr : ops) {
		operation *opn = dynamic_cast <operation *> (pr.second);

		if (table.count(pr.first.first))
			table[pr.first.first].push_back({pr.first.second, pr.second});
		else
			table.insert(std::pair <std::string, std::vector <std::pair <signature, Token *>>> {pr.first.first, std::vector <std::pair <signature, Token *>> {{pr.first.second, pr.second}}});
	}
}

Barn &Barn::operator=(const Barn &other)
{
	if (this != &other) {
		__var_table = other.__var_table;
		__ftr_table = other.__ftr_table;

		for (auto pr : other.ops)
			ops.push_back({pr.first, pr.second->copy()});
		
		for (auto pr : ops) {
			operation *opn = dynamic_cast <operation *> (pr.second);

			if (table.count(pr.first.first))
				table[pr.first.first].push_back({pr.first.second, pr.second});
			else
				table.insert(std::pair <std::string, std::vector <std::pair <signature, Token *>>> {pr.first.first, ::std::vector <::std::pair <signature, Token *>> {{pr.first.second, pr.second}}});
		}
	}

	return *this;
}

Barn::~Barn()
{
	for (auto pr : ops)
		delete pr.second;
}

bool Barn::present(const std::string &str) const
{
	auto itr = std::find_if(ops.begin(), ops.end(),
		[&](const ::std::pair <ID, Token *> &pr) {
			return pr.first.first == str;
		}
	);

	return itr != ops.end();
}

void Barn::put(Variable var)
{
	if (__var_table.count(var.symbol()))
		__var_table[var.symbol()] = var;
	else
		__var_table.insert(std::make_pair(var.symbol(), var));
}

void Barn::put(Registrable reg)
{
	if (__reg_table.find(reg.str()) != __reg_table.end())
		__reg_table[reg.str()] = reg;
	else
		__reg_table.insert(std::make_pair(reg.str(), reg));
}

void Barn::put(algorithm alg)
{
	if (__alg_table.find(alg.symbol()) != __alg_table.end())
		__alg_table[alg.symbol()] = alg;
	else
		__alg_table.insert(std::make_pair(alg.symbol(), alg));
}

/*
template <class T, class U>
template <class A>
void Barn <T, U> ::put(const ::std::string &str, A x)
{
	put(Variable(str, x));
} */

void Barn::put(Token *tptr, const std::string &str)
{
	// Leave the passed pointer alone, and copy it instead
	put(Variable(tptr->copy(), str));
}

Variable &Barn::retrieve_variable(const std::string &str)
{
	return __var_table[str];
}

Token *Barn::get(const std::string &str)
{
	// Prioritize algorithms
	if (__alg_table.count(str))
		return __alg_table[str].copy();
	
	if (__reg_table.count(str))
		return __reg_table[str].copy();
	
	if (__var_table.count(str))
		return __var_table[str].copy();
	
	if (__ftr_table.count(str))
		return __ftr_table[str].copy();
	
	return nullptr;
}

Token *Barn::compute(const std::string &str, const std::vector <Token *> &vals) const
{
	std::vector <std::type_index> sig;

	for (Token *tptr : vals)
		sig.push_back(typeid(*tptr));

	Token *tptr = nullptr;

	std::vector <std::pair <signature, Token *>> *siglist = &table[str];

	size_t sz = sig.size();
	for (auto itr = siglist->begin(); itr != siglist->end(); itr++) {
		if (itr->first.size() == sz) {
			bool ps = true;
			
			for (size_t i = 0; i < sz; i++) {					
				if (sig[i] != itr->first[i]) {
					ps = false;

					break;
				}
			}

			if (ps) {
				tptr = itr->second;
				
				break;
			}
		}
	}
	
	if (tptr) {
		operation *optr = dynamic_cast <operation *> (tptr);

		return (*optr)(vals);
	} else {
		std::ostringstream oss;

		oss << "Unknown overload (";

		for (size_t i = 0; i < sig.size(); i++) {
			oss << types::symbol(sig[i]);
			
			if (i < sig.size() - 1)
				oss << ", ";
		}

		oss << ") for operation \"" << str << "\". " <<
			overloads(str);

		throw unknown_operation_overload_exception(oss.str());
	}

	return nullptr;
}

std::string Barn::overloads(const std::string &str) const
{
	std::vector <::std::vector <::std::type_index>> loads;
	for (auto itr = ops.begin(); itr != ops.end(); itr++) {
		if (itr->first.first == str)
			loads.push_back(itr->first.second);
	}

	std::ostringstream oss;

	oss << "Available overloads for \"" << str << "\": {";

	for (size_t k = 0; k < loads.size(); k++) {
		oss << "(";

		for (size_t i = 0; i < loads[k].size(); i++) {
			oss << types::symbol(loads[k][i]);
			
			if (i < loads[k].size() - 1)
				oss << ", ";
		}

		oss << ")";

		if (k < loads.size() - 1)
			oss << ", ";
	}

	oss << "}";

	return oss.str();
}

void Barn::list() const
{
	std::cout << "\tVariables:" << std::endl;
	for (auto spr : __var_table)
		std::cout << "\t\t" << spr.second.str() << std::endl;
	
	std::cout << "\tFunctions:" << std::endl;
	for (auto spr : __ftr_table)
		std::cout << spr.second.str() << std::endl;
}

void Barn::list_registered(std::string file) const
{
	printf("Symbols recorded in %s:\n", file.c_str());
	for (auto spr : __reg_table)
		std::cout << "\t" << spr.second.str() << std::endl;
}

void Barn::print(bool show_ops) const
{
	std::cout << ::std::string(50, '-') << ::std::endl;
	std::cout << "Variables:" << ::std::endl;
	std::cout << ::std::string(50, '-') << ::std::endl;

	for (auto spr : __var_table)
		std::cout << spr.second.str() << std::endl;

	std::cout << std::string(50, '-') << std::endl;
	std::cout << "Functions:" << std::endl;
	std::cout << std::string(50, '-') << std::endl;

	for (auto spr : __ftr_table)
		std::cout << spr.second.str() << std::endl;

	::std::cout << ::std::string(50, '-') << ::std::endl;
	::std::cout << "Reg Table:" << ::std::endl;
	::std::cout << ::std::string(50, '-') << ::std::endl;

	for (auto spr : __reg_table)
		std::cout << spr.second.str() << std::endl;
	
	::std::cout << ::std::string(50, '-') << ::std::endl;
	::std::cout << "Algorithms [" << __alg_table.size() << "]" << ::std::endl;
	::std::cout << ::std::string(50, '-') << ::std::endl;

	for (auto spr : __alg_table)
		::std::cout << spr.second.str() << ::std::endl;

	if (show_ops) {
		for (auto pr : ops) {
			::std::cout << "op: " << pr.second->str() << " @ " <<
				pr.second << ::std::endl;
		}

		::std::cout << "######################################################" << ::std::endl;

		for (auto itr : table) {
			::std::cout << "Str: " << itr.first << " @ Size: " << itr.second.size() << " @ 2nd Size: " << table[itr.first].size() << ::std::endl;

			for (auto pr : itr.second)
				::std::cout << "\t" << pr.second->str() << ::std::endl;
		}
	}
}

}
