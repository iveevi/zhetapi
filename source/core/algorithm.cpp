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

// TODO: remove the alg parameter
algorithm::algorithm(
		const std::string &ident,
		const std::string &alg,
		const std::vector <std::string> &args,
		const node_manager &compiled)
		: _ident(ident), _args(args),
		_alg(alg), _compiled(compiled) {}


// Splitting equalities
static std::vector <std::string> split_assignment_chain(std::string str)
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

static void generate_statement(
		Engine *engine,
		std::string str,
		node_manager &rnm,
		Args args,
		std::set <std::string> &pardon)
{
	// Skip comments
	if (str[0] == '#')
		return;

	std::vector <std::string> tmp = split_assignment_chain(str);
	
	size_t split_size = tmp.size();
	if (split_size > 1) {
		node_manager eq;

		// Right terms are all l-values
		for (size_t i = 0; i < split_size - 1; i++) {
			// Keep track of new variables
			if (!engine->get(tmp[i]))
				pardon.insert(pardon.begin(), tmp[i]);

			// Add lvalue to the chain regardless of its previous presence
			eq.append(node(new lvalue(tmp[i]), l_lvalue));
		}

		// Only node to actually be computed (as an l-value)
		node_manager nm;
		try {
			nm = node_manager(engine, tmp[split_size - 1], args);
		} catch (const node_manager::undefined_symbol &e) {
			std::cerr << "[LINE] Unknown symbol \"" << e.what() << "\"" << std::endl;

			exit(-1);
		}

		eq.append_front(nm);
		eq.set_label(l_assignment_chain);

		rnm.append(eq);
	} else {		
		// All functions and algorithms are stored in engine
		node_manager mg;
		
		try {
			mg = node_manager(engine, str, args, pardon);
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

static int parse_parenthesized(const std::string &code, size_t &i, std::string &parenthesized)
{
	char c;

	size_t n = code.length();
	while ((i < n) && isspace(c = code[i++]));
		// __lineup(c);

	if (c != '(')
		return -1;

	int level = 0;
	while ((i < n) && (c = code[i++])) {
		if (c == '(') {
			level++;
		} else if (c == ')') {
			if (!level)
				break;
			
			level--;
		}

		parenthesized += c;
	}

	return 0;
}

static int extract_block(const std::string &code, size_t &i, std::string &block)
{
	char c;
	
	// __skip_space();
	size_t n = code.length();
	while ((i < n) && isspace(c = code[i++]));

	if (c == '{') {
		int level = 0;
		while ((i < n) && (c = code[i++])) {
			if (c == '{') {
				level++;
			} else if (c == '}') {
				if (!level)
					break;
				
				level--;
			}

			block += c;
		}
	} else {
		i--;
		while ((i < n) && (c = code[i++]) != '\n')
			block += c;

		// __lineup(c);
	}

	return 0;
}

// TODO: add a handler class
static void check_keyword(
		const std::string &code,
		size_t &i,
		std::string &keyword,
		node_manager &rnm,
		Engine *engine,
		Args args,
		std::set <std::string> &pardon)
{
	std::string parenthesized;
	std::string block;
	std::string lname;

	if (keyword == "if") {
		if (parse_parenthesized(code, i, parenthesized)) {
			printf("Syntax error at line %lu: missing parenthesis after an if\n", 0L);
			exit(-1);
		}

		node_manager condition(engine, parenthesized, args, pardon);

		extract_block(code, i, block);

		node_manager nmblock = compile_block(engine, block + "\n", args, pardon);

		node_manager ifcond;

		ifcond.append(condition);
		ifcond.append(nmblock);

		ifcond.set_label(l_if_branch);

		rnm.append(ifcond);
		
		keyword.clear();
	}

	if (keyword == "elif") {
		if (parse_parenthesized(code, i, parenthesized)) {
			printf("Syntax error at line %lu: missing parenthesis after an elif\n", 0L);
			exit(-1);
		}

		node_manager condition(engine, parenthesized, args, pardon);

		extract_block(code, i, block);

		node_manager nmblock = compile_block(engine, block + "\n", args, pardon);

		node_manager elifcond;

		elifcond.append(condition);
		elifcond.append(nmblock);

		elifcond.set_label(l_elif_branch);

		rnm.append(elifcond);
		
		keyword.clear();
	}

	if (keyword == "else") {
		extract_block(code, i, block);

		node_manager nmblock = compile_block(engine, block + "\n", args, pardon);

		node_manager elsecond;

		elsecond.append(nmblock);

		elsecond.set_label(l_else_branch);

		rnm.append(elsecond);
		
		keyword.clear();
	}

	if (keyword == "while") {
		if (parse_parenthesized(code, i, parenthesized)) {
			printf("Syntax error at line %lu: missing parenthesis after an while\n", 0L);
			exit(-1);
		}

		node_manager condition(engine, parenthesized, args, pardon);

		extract_block(code, i, block);

		node_manager nm_block = compile_block(engine, block + "\n", args, pardon);

		node_manager nm_while;

		nm_while.append(condition);
		nm_while.append(nm_block);

		nm_while.set_label(l_while_loop);

		rnm.append(nm_while);
		
		keyword.clear();
	}

	if (keyword == "break") {
		node_manager nm_break;

		nm_break.set_label(l_break_loop);

		rnm.append(nm_break);

		keyword.clear();
	}

	// Acknowledge the "alg" keyword but throw an error
}

node_manager compile_block(
		Engine *engine,
		const std::string &code,
		Args args,
		std::set <std::string> &pardon)
{
	// Setup the compiled
	node_manager compiled;

	// using namespace std;
	// cout << "CODE BLOCK: \"" << code << "\"" << endl;

	// Use the definition line number
	bool quoted = false;
	int paren = 0;
	
	std::string tmp;

	size_t i = 0;
	
	size_t n = code.length();

	char c;
	while ((i < n) && (c = code[i++])) {
		if (!quoted) {
			if (c == '\"')
				quoted = true;
			if (c == '(')
				paren++;
			if (c == ')')
				paren--;
			
			if (c == '\n' || (!paren && c == ',')) {
				if (!tmp.empty()) {
					generate_statement(engine, tmp, compiled, args, pardon);

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

		check_keyword(code, i, tmp, compiled, engine, args, pardon);
	}

	compiled.add_args(args);
	compiled.set_label(l_sequential);
	compiled.compress_branches();

	/* using namespace std;
	cout << "compiled:" << endl;
	compiled.print(); */

	return compiled;
}

void algorithm::compile(Engine *engine)
{
	_compiled = compile_block(engine, _alg, _args, _pardon);
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

// Symbol
const std::string &algorithm::symbol() const
{
	return _ident;
}

void algorithm::print() const
{
	std::cout << "COMPILED:" << std::endl;
	_compiled.print();
}

bool algorithm::empty() const
{
	return _compiled.empty();
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
