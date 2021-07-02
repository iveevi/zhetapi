#include "../../engine/lang/parser.hpp"
#include "../../engine/lang/error_handling.hpp"
#include "../../engine/core/node_manager.hpp"
#include "../../engine/core/common.hpp"

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

void run_normal(const std::string &cache, Engine *context, bool line)
{
	node_manager mg;

	try {
		mg = node_manager(context, cache);

		// std::cout << "cache = \"" << cache << "\"" << std::endl;
		// std::cout << "MG:" << std::endl;
		// mg.print();

		// Print on non-null result if inline interpreting
		Token *tptr = mg.value(context);

		// TODO: add a real str method
		if (line && tptr)
			std::cout << "\t" << tptr->dbg_str() << std::endl;
	} catch (const Engine::unknown_op_overload &e)  {
		std::cerr << "FIXME: Error evaluating \'" << cache << "\'\n"
			<< e.what() << std::endl;
		exit(-1);
	} catch (const node_manager::undefined_symbol &e) {
		// TODO: include line number
		symbol_error_msg(e.what(), "", context);

		exit(-1);
	} catch (const Token::unknown_attribute &e) {
		std::cerr << "FIXME: Error: unknown attribute \"" << e.what() << "\"" << std::endl;
		// cout << "Error at line " << line
		//	<< ": " << e.what() << endl;

		exit(-1);
	} /* catch (...) {
		throw std::runtime_error(
			"Caught unknown exception in zhetapi::run..."
		);
	} */
}

// TODO: clean
void run_assignment(const Args &veq, Engine *ctx, bool line)
{
	// TODO: is this overkill?
	Pardon pardon;

	node_manager chain = cc_run_assignment(veq, ctx, {}, pardon);
	/* std::cout << "chain:" << std::endl;
	chain.print(); */

	Token *tptr = chain.value(ctx);

	// TODO: add a real str method
	if (line && tptr)
		std::cout << "\t" << tptr->dbg_str() << std::endl;
}

void run(const std::string &cache, Engine *context, bool line)
{
	Args veq = eq_split(cache);

	if (veq.size() > 1)
		return run_assignment(veq, context, line);

	run_normal(cache, context, line);
}

int parse_global(Feeder *feeder, Engine *context, const Args &idirs, const char *linted_version, bool line)
{
	/* State of parsing: not static to allow multiple threads to parse
	 * (possibly) different sources at the same time.
	 *
	 * Possible turn into an external struct
	 */
	State state;

	// Set idirs and lver
	state.idirs = idirs;
	state.lver = linted_version;

	char c;
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

			run(state.cached, context, line);

			// Clear cached
			state.cached.clear();

			// Skip the rest of the loop
			continue;
		}

		// Unterminated parenthesis
		if (c == '\n' && (state.parenthesis
				&& feeder->prev() != ','))
			throw bad_line(state.cached);

		if (!isspace(c))
			state.cached += c;

		OpZ *opz = check_keyword(state.cached, feeder, context, &state);

		if (opz)
			return opz->get();

		// cout << "cached = " << state.cached << endl;
	}

	// TODO: dont forget to check parenthesis and such later
	if (!state.cached.empty()) {
		run(state.cached, context, line);

		// Clear cached
		state.cached.clear();
	}

	return 0;
}

}
