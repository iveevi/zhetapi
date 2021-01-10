#include <core/algorithm.hpp>
#include <barn.hpp>
#include <function.hpp>

namespace zhetapi {

// Constructors
algorithm::algorithm() {}

algorithm::algorithm(std::string ident,
		const std::vector <std::string> &args,
		const std::string &alg) : __ident(ident),
		__args(args), __alg(alg) {}

// Executing the function
Token *algorithm::execute(const Barn &barn, const std::vector <Token *> &args)
{
	Barn *cpy = new Barn(barn);

	// For now, no default arguments or overloads
	assert(args.size() == __args.size());

	size_t n = __args.size();
	for (size_t i = 0; i < n; i++)
		cpy->put(args[i], __args[i]);

	// Use the definition line number
	bool quoted = false;
	int paren = 0;
	
	std::string tmp;

	size_t i = 0;
	
	n = __alg.length();

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
					execute(cpy, tmp);

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

	// Return the "return" value instead of nullptr
	return nullptr;
}

Token *algorithm::execute(Barn *barn, std::string str)
{
	// Skip comments
	if (str[0] == '#')
		return nullptr;

	std::vector <std::string> tmp = split(str);
	
	size_t tsize = tmp.size();
	if (tsize > 1) {
		zhetapi::Token *tptr = nullptr;
		
		try {
			zhetapi::node_manager mg(tmp[tsize - 1], barn);

			tptr = mg.value();
		} catch (...) {}

		for (int i = tsize - 2; i >= 0; i--) {
			std::string ftr = tmp[i] + " = " + tmp[tsize - 1];

			try {
				zhetapi::Function f = ftr;

				barn->put(f);
			} catch (...) {
				barn->put(tptr, tmp[i]);
			}
		}
		
		delete tptr;
	} else {		
		// All functions and algorithms are stored in barn
		node_manager mg;
		
		try {
			mg = node_manager(str, barn);
		} catch (node_manager::undefined_symbol e) {
			std::cout << "Error at line " << 0
				<< ": undefined symbol \""
				<< e.what() << "\"" << std::endl;

			exit(-1);
		}

		/* std::cout << "mg:" << std::endl;
		mg.print(); */

		// "Execute" the statement
		return mg.value(barn);
	}

	return nullptr;
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
	return new algorithm(__ident, __args, __alg);
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
