#include <lang/parser.hpp>
#include <lang/error_handling.hpp>
#include <core/node_manager.hpp>
#include <core/common.hpp>

namespace zhetapi {

// TODO: another file
State::State(bool bl) : alg(bl) {}

void State::check_nest(char c) {
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

bool State::is_nested() {
	return parenthesis || braces || brackets;
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
		} catch (...) {
			if (pe) {
				symbol_error_msg(us.what(), "", context);
				exit(-1);
			}

			context->put(veq[i], tptr);
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
	}

	return 0;
}

}
