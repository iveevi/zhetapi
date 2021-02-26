#include <core/algorithm.hpp>
#include <barn.hpp>
#include <function.hpp>

namespace zhetapi {

// Constructors
algorithm::algorithm() {}

algorithm::algorithm(
		const std::string &ident,
		const std::string &alg,
		const std::vector <std::string> &args)
		: __ident(ident), __args(args),
		__alg(alg) {}

algorithm::algorithm(
		const std::string &ident,
		const std::string &alg,
		const std::vector <std::string> &args,
		const node_manager &compiled)
		: __ident(ident), __args(args),
		__alg(alg), __compiled(compiled) {}

void algorithm::compile(Barn *barn)
{
	using namespace std;
	cout << "Compiling..." << endl;
	for (auto str : __args)
		cout << "args = " << str << endl;

	// Use the definition line number
	bool quoted = false;
	int paren = 0;
	
	std::string tmp;

	size_t i = 0;
	
	size_t n = __alg.length();

	char c;
	while ((i < n) && (c = __alg[i++])) {
		if (!quoted) {
			if (c == '\"')
				quoted = true;
			if (c == '(')
				paren++;
			if (c == ')')
				paren--;
			
			if (c == '\n' || (!paren && c == ',')) {
				if (!tmp.empty()) {
					generate(barn, tmp, __compiled);

					tmp.clear();
				}
			} else if (!isspace(c)) {
				tmp += c;
			}
		} else {
			if (c == '\"')
				quoted = false;
			
			tmp += c;
		}
	}

	using namespace std;
	cout << string(50, '=') << endl;
	cout << "ARGUMENTS" << endl;
	__compiled.tree().__label = l_sequential;
	__compiled.add_args(__args);
	cout << string(50, '=') << endl;

	cout << "finished compiling, full alg: " << this << endl;
	__compiled.print(true);
}

void algorithm::generate(Barn *barn, std::string str, node_manager &rnm)
{
	// Skip comments
	if (str[0] == '#')
		return;

	std::vector <std::string> tmp = split(str);
	
	size_t tsize = tmp.size();
	if (tsize > 1) {
		// Ignore assignment for now

		/*
		std::vector <node> order;

		zhetapi::Token *tptr = nullptr;
		
		try {
			zhetapi::node_manager mg(tmp[tsize - 1], barn);

			tptr = mg.value();

			order.push_back(mg.tree());
		} catch (...) {}

		for (int i = tsize - 2; i >= 0; i--) {
			std::string ftr = tmp[i] + " = " + tmp[tsize - 1];

			try {
				zhetapi::Function f = ftr;

				barn->put(f);

				order.push_back(node(&f, {}));
			} catch (...) {
				barn->put(tptr, tmp[i]);

				order.push_back(node(new Operand <std::string> (tmp[i]), {}));
			}
		}

		node n(nullptr, l_assign_chain, order);

		rnm.append(n);
		
		delete tptr; */
	} else {		
		// All functions and algorithms are stored in barn
		node_manager mg;
		
		try {
			mg = node_manager(str, __args, barn);
		} catch (const node_manager::undefined_symbol &e) {
			std::cout << "Error at line " << 0
				<< ": undefined symbol \""
				<< e.what() << "\"" << std::endl;

			exit(-1);
		}

		rnm.append(mg);

		// "Execute" the statement
		// return mg.value(barn);
	}

	// return nullptr;
}

// Executing the function
Token *algorithm::execute(Barn *barn, const std::vector <Token *> &args)
{
	// Ignore arguments for now
	if (__compiled.empty())
		compile(barn);

	return __compiled.substitute_and_seq_compute(barn, args);
}

// Splitting equalities
std::vector <std::string> algorithm::split(std::string str)
{
	bool quoted = false;

	char pc = 0;

	std::vector <std::string> out;

	size_t n = str.length();

	std::string tmp;
	for (size_t i = 0; i < n; i++) {
		if (!quoted) {
			bool ignore = false;

			if (pc == '>' || pc == '<' || pc == '!'
				|| (i > 0 && str[i - 1] == '='))
				ignore = true;
			
			if (!ignore && str[i] == '=') {
				if (i < n - 1 && str[i + 1] == '=') {
					tmp += "==";
				} else if (!tmp.empty()) {
					out.push_back(tmp);

					tmp.clear();
				}
			} else {
				if (str[i] == '\"')
					quoted = true;
				
				tmp += str[i];
			}
		} else {
			if (str[i] == '\"')
				quoted = false;
			
			tmp += str[i];
		}

		pc = str[i];
	}

	if (!tmp.empty())
		out.push_back(tmp);
	
	/* cout << "split:" << endl;
	for (auto s : out)
		cout << "\ts = " << s << endl; */

	return out;
}

// Symbol
const std::string &algorithm::symbol() const
{
	return __ident;
}

// Virtual functions
Token::type algorithm::caller() const
{
	return Token::alg;
}

Token *algorithm::copy() const
{
	return new algorithm(__ident, __alg, __args, __compiled);
}

std::string algorithm::str() const
{
	return __ident;
}

bool algorithm::operator==(Token *tptr) const
{
	algorithm *alg = dynamic_cast <algorithm *> (tptr);

	if (alg == nullptr)
		return false;

	return alg->__ident == __ident;
}

}
