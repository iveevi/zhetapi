#include <lang/parser.hpp>
#include <lang/error_handling.hpp>
#include <core/node_manager.hpp>
#include <core/common.hpp>

namespace zhetapi {

// Inserting variables
static void run_assignment(const Args &veq, Engine *ctx, Module *module)
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

		tptr = new Function(ftr, ctx);

		module->add(fout[0], tptr);
		ctx->put(fout[0], tptr);
	} else {
		tptr = (node_manager(ctx, veq[n - 1])).value(ctx);

		module->add(fout[0], tptr);
		ctx->put(fout[0], tptr);
	}

	for (int i = n - 3; i >= 0; i--) {
		Args kout = get_args(veq[i]);
		Args kargs(kout.begin() + 1, kout.end());

		if (kout.size() > 1 && is_valid_ident(kout[0])) {
			if (!in_args(fargs, kargs))
				throw args_mismatch(veq[i]);

			std::string ftr = veq[i] + " = " + veq[n - 1];

			// Update the token with functions
			tptr = new Function(ftr, ctx);

			module->add(kout[0], tptr);
			ctx->put(kout[0], tptr);
		} else if (kout.size() > 0 && is_valid_ident(kout[0])) {
			module->add(kout[0], tptr);
			ctx->put(kout[0], tptr);
		} else {
			throw bad_identifier(veq[i]);
		}
	}
}

// TODO: make sure state->branch gets reset
static void mdl_if(Feeder *feeder,
		Engine *context,
		Module *module,
		State *state)
{
	// Save end
	char end = feeder->get_end();

	// Update the state
	state->branch = true;

	char c;
	while ((c = feeder->feed()) != '(');

	// TODO: add a skip until for characters
	feeder->skip_until((c == '{') ? "}" : "\n");
}

static void mdl_elif(Feeder *feeder,
		Engine *context,
		Module *module,
		State *state)
{
	// Save end
	char end = feeder->get_end();

	// TODO: throw exception
	if (!state->branch)
		throw bad_elif();

	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	while (isspace(c = feeder->feed()));

	// TODO: add a skip until for characters
	feeder->skip_until((c == '{') ? "}" : "\n");
}

static void mdl_else(Feeder *feeder,
		Engine *context,
		Module *module,
		State *state)
{
	// Save end
	char end = feeder->get_end();

	// Throw exception
	if (!state->branch)
		throw bad_else();

	char c;

	while (isspace(c = feeder->feed()));

	// Skip block
	feeder->skip_until((c == '{') ? "}" : "\n");

	// End the current branch
	state->branch = false;
}

static void mdl_while(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	// Save end
	char end = feeder->get_end();

	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	while (isspace(c = feeder->feed()));

	// Skip block
	feeder->skip_until((c == '{') ? "}" : "\n");
}

static void mdl_for(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	// Save end
	char end = feeder->get_end();

	char c;
	while ((c = feeder->feed()) != '(');
	std::string paren = feeder->extract_parenthesized();
	while (isspace(c = feeder->feed()));
	if (c != '{')
		feeder->backup(1);
	feeder->set_end((c == '{') ? '}' : '\n');

	// Reset terminal
	feeder->set_end(end);
}

static void mdl_break(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	throw global_break();
}

static void mdl_continue(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	throw global_continue();
}

static void mdl_alg(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	// Save end
	char end = feeder->get_end();

	std::pair <std::string, Args> sig = feeder->extract_signature();

	char c;
	while (isspace(c = feeder->feed()));
	if (c != '{')
		feeder->backup(1);

	feeder->set_end((c == '{') ? '}' : '\n');

	Pardon pardon;
	node_manager nbody = cc_parse(feeder, ctx, sig.second, pardon);

	nbody.add_args(sig.second);
	nbody.set_label(l_sequential);
	nbody.compress_branches();

	algorithm alg(sig.first, "", sig.second, nbody);

	ctx->put(alg);
	module->add({sig.first, alg.copy()});

	// Reset terminal
	feeder->set_end(end);
}

static void mdl_return(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	// TODO: Allow multiline if the user add '\'
	char c;
	while ((c = feeder->feed()) != '\n' && c != EOF);
}

static void mdl_global(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	// TODO: Allow multiline if the user add '\'
	char c;

	std::string line;
	while ((c = feeder->feed()) != '\n' && c != EOF)
		line += c;

	Args vcomma = comma_split(line);
	for (const std::string &as : vcomma) {
		Args veq = eq_split(as);

		if (veq.size() > 1)
			run_assignment(veq, ctx, module);
	}
}

// Static?
static void mdl_keyword(std::string &cache,
		Feeder *feeder,
		Engine *context,
		Module *module,
		State *state)
{
	using Processor = std::function <void (Feeder *, Engine *, Module *, State *)>;
	static const Symtab <Processor> keywords {
		{"if", mdl_if},
		{"elif", mdl_elif},
		{"else", mdl_else},
		{"while", mdl_while},
		{"for", mdl_for},
		{"break", mdl_break},
		{"continue", mdl_continue},
		{"alg", mdl_alg},
		{"return", mdl_return},
		{"global", mdl_global}
		// TODO: Import
		// TODO: global keyword
	};

	if (keywords.find(cache) == keywords.end())
		return;

	// Execute and clear the cache
	(keywords.at(cache))(feeder, context, module, state);
	cache.clear();
}

void mdl_parse(Feeder *feeder, Engine *context, Module *module)
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

			// run(state.cached, context);

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

		mdl_keyword(state.cached, feeder, context, module, &state);

		/* if (opz)
			return opz->get();

		// cout << "cached = " << state.cached << endl; */
	}

	/* TODO: dont forget to check parenthesis and such later
	if (!state.cached.empty()) {
		run(state.cached, context);

		// Clear cached
		state.cached.clear();
	} */
}

}
