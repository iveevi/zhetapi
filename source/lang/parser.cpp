#include <lang/parser.hpp>
#include <lang/error_handling.hpp>
#include <core/node_manager.hpp>
#include <core/common.hpp>

namespace zhetapi {

// TODO: another file
State::State(bool bl) : alg(bl) {}

void State::check_nest(char c)
{
	// make sure all are positive
	switch (c) {
	case '(':
		parenthesis++;
		break;
	case ')':
		parenthesis--;
		break;
	case '{':
		braces++;
		break;
	case '}':
		braces--;
		break;
	case '[':
		brackets++;
		break;
	case ']':
		brackets--;
		break;
	}
}

bool State::is_nested()
{
	return parenthesis
		|| braces
		|| brackets;
}

// TODO: clean
static std::vector <std::string> eq_split(const std::string &str)
{
	// TODO: dont forget about op=
	// TODO: if possible, split in the first pass
	bool quoted = false;

	char pc = 0;

	std::vector <std::string> out;
	size_t n;

	n = str.length();

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

	return out;
}

static void run_normal(const std::string &cache, Engine *context)
{
	node_manager mg;

	try {
		mg = node_manager(context, cache);

		mg.value(context);
	} catch (const Engine::unknown_op_overload &e)  {
		std::cout << "err evaluating \'" << cache << "\'\n" << e.what() << std::endl;
		exit(-1);
	} catch (const node_manager::undefined_symbol &e) {
		// TODO: include line number
		symbol_error_msg(e.what(), "", context);

		exit(-1);
	} catch (const Token::unknown_attribute &e) {
		// cout << "Error at line " << line
		//	<< ": " << e.what() << endl;

		exit(-1);
	} catch (...) {
		throw std::runtime_error(
			"Caught unknown exception in zhetapi::run..."
		);
	}
}

// TODO: clean
static void run_assignment(const std::vector <std::string> &veq, Engine *context)
{
	Token *tptr = nullptr;

	node_manager::undefined_symbol us("");
	bool pe = false;

	size_t n = veq.size();
	try {
		node_manager mg(context, veq[n - 1]);

		tptr = mg.value(context);
	} catch (const Engine::unknown_op_overload &e)  {
		// TODO: color errors
		std::cout << "err: " << e.what() << std::endl;
		exit(-1);
	} catch (const node_manager::undefined_symbol &e) {
		us = e;
		pe = true;
	}

	for (int i = n - 2; i >= 0; i--) {
		std::string ftr = veq[i] + " = " + veq[n - 1];

		try {
			Function f(ftr, context);

			context->put(f);
		} catch (const node_manager::undefined_symbol &e) {
			symbol_error_msg(e.what(), "", context);
			exit(-1);
		} catch (const Function::invalid_definition &e) {
			context->put(veq[i], tptr);
		} catch (...) {
			// TODO: fatal error Something else
			throw;
		}
	}

	delete tptr;
}

void run(const std::string &cache, Engine *context)
{
	std::vector <std::string> veq = eq_split(cache);

	if (veq.size() > 1)
		return run_assignment(veq, context);

	run_normal(cache, context);
}

// TODO: move
static node_manager cc_run_normal(const std::string &cache, Engine *context)
{
	node_manager mg;

	try {
		mg = node_manager(context, cache);
	} catch (const Engine::unknown_op_overload &e)  {
		std::cout << "err evaluating \'" << cache << "\'\n" << e.what() << std::endl;
		exit(-1);
	} catch (const node_manager::undefined_symbol &e) {
		// TODO: include line number
		symbol_error_msg(e.what(), "", context);

		exit(-1);
	} catch (const Token::unknown_attribute &e) {
		// cout << "Error at line " << line
		//	<< ": " << e.what() << endl;

		exit(-1);
	} catch (...) {
		throw std::runtime_error(
			"Caught unknown exception in zhetapi::run..."
		);
	}

	return mg;
}

// TODO: clean and move
static node_manager cc_run_assignment(const std::vector <std::string> &veq, Engine *ctx)
{
	node_manager out;

	// TODO: assign functions?

	// Right terms are all l-values
	size_t n = veq.size();
	for (size_t i = 0; i < n - 1; i++) {
		// Keep track of new variables
		if (!engine->get(veq[i]))
			pardon.insert(pardon.begin(), veq[i]);

		// Add lvalue to the chain regardless of its previous presence
		out.append(node(new lvalue(tmp[i]), l_lvalue));
	}

	// Only node to actually be computed (as an l-value)
	node_manager nm;
	try {
		// TODO: take args
		// nm = node_manager(ctx, veq[n - 1], args, pardon);
		nm = node_manager(ctx, veq[n - 1]);
	} catch (const node_manager::undefined_symbol &e) {
		symbol_error_msg(e.what(), tmp[n - 1], ctx);

		exit(-1);
	}

	out.append_front(nm);
	out.set_label(l_assignment_chain);
	
	return out;
}

node_manager cc_run(const std::string &cache, Engine *context)
{
	std::vector <std::string> veq = eq_split(cache);

	if (veq.size() > 1)
		return cc_run_assignment(veq, context);

	return cc_run_normal(cache, context);
}

int parse_global(const std::string &source, Engine *context)
{
	Feeder *feeder = new StringFeeder(source);

	return parse_global(feeder, context);
}

int parse_global(Feeder *feeder, Engine *context)
{
	/* State of parsing: not static to allow multiple threads to parse
	 * (possibly) different sources at the same time.
	 *
	 * Possible turn into an external struct
	 */
	struct State state;

	char c;
	
	// TODO: remove using
	using namespace std;
	while ((c = feeder->feed()) != EOF) {
		// Check for commented line
		// TODO: another function
		if (c == '/') {
			// Read until end of line
			if (feeder->peek() == '/') {
				feeder->skip_line();

				// Skip the rest of the loop
				continue;
			}

			// Read until reaches end of comment
			if (feeder->peek() == '*') {
				feeder->skip_until("*/");

				// Skip the rest of the loop
				continue;
			}
		}

		// Extract string
		if (c == '\"') {
			state.cached += '\"' + feeder->extract_quote() + '\"';

			// Skip the rest of the loop
			continue;
		}

		// Check for parenthesis, braces or brackets
		state.check_nest(c);

		// TODO: check for parenthesis (even for newline)
		if (!state.is_nested() && (c == '\n' || c == ',')
				&& !state.cached.empty()) {
			// cout << "cached string \"" << state.cached << "\" is ready..." << endl;

			// TODO: process cached
			run(state.cached, context);

			// Clear cached
			state.cached.clear();

			// Skip the rest of the loop
			continue;
		}

		if (!isspace(c))
			state.cached += c;
		
		check_keyword(state.cached, feeder, context, &state);
		// cout << "cached = " << state.cached << endl;
	}

	return 0;
}

node_manager cc_parse(Feeder *feeder, Engine *ctx)
{
	node_manager out;

	struct State state;

	char c;
	
	// TODO: remove using
	using namespace std;
	while ((c = feeder->feed()) != EOF) {
		// Check for commented line
		// TODO: another function
		if (c == '/') {
			// Read until end of line
			if (feeder->peek() == '/') {
				feeder->skip_line();

				// Skip the rest of the loop
				continue;
			}

			// Read until reaches end of comment
			if (feeder->peek() == '*') {
				feeder->skip_until("*/");

				// Skip the rest of the loop
				continue;
			}
		}

		// Extract string
		if (c == '\"') {
			state.cached += '\"' + feeder->extract_quote() + '\"';

			// Skip the rest of the loop
			continue;
		}

		// Check for parenthesis, braces or brackets
		state.check_nest(c);

		// TODO: check for parenthesis (even for newline)
		if (!state.is_nested() && (c == '\n' || c == ',')
				&& !state.cached.empty()) {
			// cout << "cached string \"" << state.cached << "\" is ready..." << endl;

			// TODO: process cached
			out.append(cc_run(state.cached, ctx));

			// Clear cached
			state.cached.clear();

			// Skip the rest of the loop
			continue;
		}

		if (!isspace(c))
			state.cached += c;
		
		// check_keyword(state.cached, feeder, context, &state);
		// cout << "cached = " << state.cached << endl;
	}

	// Finalize the tree
	out.set_label(l_sequential);
	out.compress_branches();

	return out;
}

}
