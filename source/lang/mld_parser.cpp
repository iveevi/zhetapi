#include <lang/parser.hpp>
#include <lang/error_handling.hpp>
#include <core/node_manager.hpp>
#include <core/common.hpp>

namespace zhetapi {

// TODO: make sure state->branch gets reset
static void mld_if(Feeder *feeder,
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

static void mld_elif(Feeder *feeder,
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

static void mld_else(Feeder *feeder,
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

static void mld_while(Feeder *feeder,
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

static void mld_for(Feeder *feeder,
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

static void mld_break(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	throw global_break();
}

static void mld_continue(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	throw global_continue();
}

static void mld_alg(Feeder *feeder,
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

static void mld_return(Feeder *feeder,
		Engine *ctx,
		Module *module,
		State *state)
{
	// TODO: Allow multiline if the user add '\'
	char c;
	while ((c = feeder->feed()) != '\n');
}

// Static?
static void mld_keyword(std::string &cache,
		Feeder *feeder,
		Engine *context,
		Module *module,
		State *state)
{
	using Processor = std::function <void (Feeder *, Engine *, Module *, State *)>;
	static const Symtab <Processor> keywords {
		{"if", mld_if},
		{"elif", mld_elif},
		{"else", mld_else},
		{"while", mld_while},
		{"for", mld_for},
		{"break", mld_break},
		{"continue", mld_continue},
		{"alg", mld_alg},
		{"return", mld_return}
		// TODO: Import
		// TODO: global keyword
	};

	if (keywords.find(cache) == keywords.end())
		return;
	
	// Execute and clear the cache
	(keywords.at(cache))(feeder, context, module, state);
	cache.clear();
}

void mld_parse(Feeder *feeder, Engine *context, Module *module)
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
		
		mld_keyword(state.cached, feeder, context, module, &state);

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