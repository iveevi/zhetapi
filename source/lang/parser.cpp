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
	} /* catch (...) {
		throw std::runtime_error(
			"Caught unknown exception in zhetapi::run..."
		);
	} */
}

// TODO: clean
static void run_assignment(const std::vector <std::string> &veq, Engine *ctx)
{
	size_t n = veq.size();

	// Get possible arguments for the first assignment
	Args fout = get_args(veq[n - 2]);
	Args fargs(fout.begin() + 1, fout.end());

	// Ensure valid identifier
	if (fout.empty() || !is_valid_ident(fout[0]))
		throw bad_identifier(veq[n - 2]);

	Token *tptr = nullptr;
	if (fout.size() > 1) {
		std::string ftr = veq[n - 2] + " = " + veq[n - 1];

		ctx->put(Function(ftr, ctx));
		tptr = ctx->get(fout[0]);
	} else {
		tptr = (node_manager(ctx, veq[n - 1])).value(ctx);

		ctx->put(fout[0], tptr);
	}

	for (int i = n - 3; i >= 0; i--) {
		Args kout = get_args(veq[i]);
		Args kargs(kout.begin() + 1, kout.end());

		if (kout.size() > 1 && is_valid_ident(kout[0])) {
			if (!in_args(fargs, kargs))
				throw args_mismatch(veq[i]);

			std::string ftr = veq[i] + " = " + veq[n - 1];
			ctx->put(Function(ftr, ctx));

			tptr = ctx->get(kout[0]);
		} else if (kout.size() > 0 && is_valid_ident(kout[0])) {
			ctx->put(kout[0], tptr);
		} else {
			throw bad_identifier(veq[i]);
		}
	}
}

void run(const std::string &cache, Engine *context)
{
	std::vector <std::string> veq = eq_split(cache);

	if (veq.size() > 1)
		return run_assignment(veq, context);

	run_normal(cache, context);
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

			run(state.cached, context);

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
		run(state.cached, context);

		// Clear cached
		state.cached.clear();
	}

	return 0;
}

}
