#include <lang/parser.hpp>
#include <lang/error_handling.hpp>
#include <core/node_manager.hpp>
#include <core/common.hpp>

namespace zhetapi {

static node_manager cc_run_normal(const std::string &cache,
		Engine *context,
		const Args &args,
		const Pardon &pardon)
{
	node_manager mg;

	try {
		mg = node_manager(context, cache, args, pardon);
	} catch (const Engine::unknown_op_overload &e)  {
		std::cerr << "err evaluating \'" << cache << "\'\n"
			<< e.what() << std::endl;
		exit(-1);
	} catch (const node_manager::undefined_symbol &e) {
		// TODO: include line number
		symbol_error_msg(e.what(), "", context);

		exit(-1);
	} catch (const Token::unknown_attribute &e) {
		std::cerr << "Error at line " << 0
			<< ": " << e.what() << std::endl;

		exit(-1);
	} catch (...) {
		throw std::runtime_error(
			"Caught unknown exception in zhetapi::run..."
		);
	}

	return mg;
}

// TODO: clean
static node_manager cc_run_assignment(const std::vector <std::string> &veq,
		Engine *ctx,
		const Args &args,
		Pardon &pardon)
{
	node_manager out;

	size_t n = veq.size();

	Args fout = get_args(veq[n - 2]);
	Args fargs(fout.begin() + 1, fout.end());
	
	if (fout.empty() || !is_valid_ident(fout[0]))
		throw bad_identifier(veq[n - 2]);

	node_manager nm;
	if (fout.size() > 1) {
		nm = node_manager(ctx, veq[n - 1],
				args_union(args, fargs),
				pardon);
	} else {
		nm = node_manager(ctx, veq[n - 1], args, pardon);
	}

	out.append(nm);

	node ftn(new lvalue(fout[0]), l_lvalue);
	if (fout.size() > 1)
		ftn.append(node(new Operand <Args> (fargs)));

	out.append(ftn);

	// Add the first
	if (!ctx->get(fout[0]))
		pardon.insert(pardon.begin(), fout[0]);

	// TODO: args need to be valid as well
	for (int i = n - 3; i >= 0; i--) {
		Args kout = get_args(veq[i]);
		Args kargs(kout.begin() + 1, kout.end());

		if (kout.size() > 1 && is_valid_ident(kout[0])) {
			if (!in_args(fargs, kargs))
				throw args_mismatch(veq[i]);

			out.append(
				node(new lvalue(kout[0]), l_lvalue,
				{
					node(new Operand <Args> (kargs))
				})
			);

			if (!ctx->get(kout[0]))
				pardon.insert(pardon.begin(), kout[0]);
		} else if (kout.size() > 0 && is_valid_ident(kout[0])) {
			out.append(node(new lvalue(kout[0]), l_lvalue));
			
			if (!ctx->get(kout[0]))
				pardon.insert(pardon.begin(), kout[0]);
		} else {
			throw bad_identifier(veq[i]);
		}
	}
	
	out.set_label(l_assignment_chain);
	
	return out;
}

node_manager cc_run(const std::string &cache,
		Engine *context,
		const Args &args,
		Pardon &pardon)
{
	std::vector <std::string> veq = eq_split(cache);

	if (veq.size() > 1)
		return cc_run_assignment(veq, context, args, pardon);

	return cc_run_normal(cache, context, args, pardon);
}

node_manager cc_parse(Feeder *feeder,
		Engine *ctx,
		const Args &args,
		Pardon &pardon)
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

		if (!state.is_nested() && (c == '\n' || c == ',')
				&& !state.cached.empty()) {
			// cout << "cached string \"" << state.cached << "\" is ready..." << endl;

			out.append(cc_run(state.cached, ctx, args, pardon));

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
		
		// check_keyword(state.cached, feeder, context, &state);
		// cout << "cached = " << state.cached << endl;
	}

	// Finalize the tree
	out.set_label(l_sequential);
	out.compress_branches();

	return out;
}

}
