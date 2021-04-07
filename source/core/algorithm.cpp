#include <core/algorithm.hpp>
#include <engine.hpp>
#include <function.hpp>

#include <core/lvalue.hpp>
#include <core/rvalue.hpp>

namespace zhetapi {

// Constructors
algorithm::algorithm() {}

algorithm::algorithm(const algorithm &other)
		: _ident(other._ident), _alg(other._alg),
		_args(other._args), _compiled(other._compiled) {}

algorithm::algorithm(
		const std::string &ident,
		const std::string &alg,
		const std::vector <std::string> &args)
		: _ident(ident), _args(args),
		_alg(alg) {}

algorithm::algorithm(
		const std::string &ident,
		const std::string &alg,
		const std::vector <std::string> &args,
		const node_manager &compiled)
		: _ident(ident), _args(args),
		_alg(alg), _compiled(compiled) {}

void algorithm::compile(Engine *engine)
{
	// Push stack
	// engine = push_and_ret_stack(engine);
	
	// Use the definition line number
	bool quoted = false;
	int paren = 0;
	
	std::string tmp;

	size_t i = 0;
	
	size_t n = _alg.length();

	char c;
	while ((i < n) && (c = _alg[i++])) {
		if (!quoted) {
			if (c == '\"')
				quoted = true;
			if (c == '(')
				paren++;
			if (c == ')')
				paren--;
			
			if (c == '\n' || (!paren && c == ',')) {
				if (!tmp.empty()) {
					generate(engine, tmp, _compiled);

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

	// Use .set_label instead
	_compiled.add_args(_args);
	_compiled.set_label(l_sequential);
	// _compiled.set_engine(engine);

	using namespace std;
	cout << "compiled:" << endl;
	_compiled.print();
	
	// Pop stack
	// engine = pop_and_del_stack(engine);
}

void algorithm::generate(Engine *engine, std::string str, node_manager &rnm)
{
	// Skip comments
	if (str[0] == '#')
		return;

	std::vector <std::string> tmp = split(str);
	
	size_t split_size = tmp.size();
	if (split_size > 1) {
		node_manager eq;

		// Right terms are all l-values
		for (size_t i = 0; i < split_size - 1; i++) {
			// Keep track of new variables
			if (!engine->get(tmp[i]))
				_pardon.insert(_pardon.begin(), tmp[i]);

			// Add lvalue to the chain regardless of its previous presence
			eq.append(node(new lvalue(tmp[i]), l_lvalue));
		}

		// Only node to actually be computed (as an l-value)
		node_manager nm(engine, tmp[split_size - 1], _args);

		eq.append_front(nm);
		eq.set_label(l_assignment_chain);

		rnm.append(eq);
	} else {		
		// All functions and algorithms are stored in engine
		node_manager mg;
		
		try {
			mg = node_manager(engine, str, _args, _pardon);
		} catch (const node_manager::undefined_symbol &e) {
			// TODO: get line number
			std::cerr << "Error at line " << 0
				<< ": undefined symbol \""
				<< e.what() << "\"" << std::endl;

			exit(-1);
		}

		rnm.append(mg);
	}
}

// Executing the function
Token *algorithm::execute(Engine *engine, const std::vector <Token *> &args)
{
	// Ignore arguments for now
	if (_compiled.empty())
		compile(engine);
	
	engine = push_and_ret_stack(engine);

	Token *tptr = _compiled.substitute_and_seq_compute(engine, args);

	engine = pop_and_del_stack(engine);

	return tptr;
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
	return _ident;
}

// Virtual functions
Token::type algorithm::caller() const
{
	return Token::alg;
}

Token *algorithm::copy() const
{
	return new algorithm(_ident, _alg, _args, _compiled);
}

std::string algorithm::str() const
{
	return _ident;
}

bool algorithm::operator==(Token *tptr) const
{
	algorithm *alg = dynamic_cast <algorithm *> (tptr);

	if (alg == nullptr)
		return false;

	return alg->_ident == _ident;
}

}
