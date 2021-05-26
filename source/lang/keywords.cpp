#include <lang/parser.hpp>
#include <lang/error_handling.hpp>
#include <core/node_manager.hpp>

namespace zhetapi {

// TODO: put elsewhere
bool is_true(Token *tptr)
{
	static const Operand <bool> true_token(true);

	return true_token == tptr;
}

// TODO: make sure state->branch gets reset
static bool check_if(Feeder *feeder,
		Engine *context,
		State *state)
{
	// Update the state
	state->branch = true;

	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	// Skip construction step or something
	node_manager nm(context, paren);

	Token *tptr = nm.value(context);

	// TODO: Add a skip whitespace
	while (isspace(c = feeder->feed()));
	if (is_true(tptr)) {
		state->bdone = true;
		if (c != '{')
			feeder->backup(1);
		
		feeder->set_end((c == '{') ? '}' : '\n');
		parse_global(feeder, context);

		// Reset terminal
		feeder->set_end();
	} else {
		// TODO: add a skip until for characters
		feeder->skip_until((c == '{') ? "}" : "\n");
	}

	return true;
}

static bool check_elif(Feeder *feeder,
		Engine *context,
		State *state)
{
	if (!state->branch) {
		std::cerr << "TODO: elif out of place error..." << std::endl;
		exit(-1);
		return false;
	}

	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	node_manager nm(context, paren);
	Token *tptr = nm.value(context);

	while (isspace(c = feeder->feed()));
	if (!state->bdone && is_true(tptr)) {
		state->bdone = true;
		if (c != '{')
			feeder->backup(1);
		
		feeder->set_end((c == '{') ? '}' : '\n');
		parse_global(feeder, context);

		// Reset terminal
		feeder->set_end();
	} else  {
		// TODO: add a skip until for characters
		feeder->skip_until((c == '{') ? "}" : "\n");
	}

	return true;
}

static bool check_else(Feeder *feeder,
		Engine *context,
		State *state)
{
	if (!state->branch) {
		std::cerr << "TODO: else out of place error..." << std::endl;
		exit(-1);
		return false;
	}

	char c;

	while (isspace(c = feeder->feed()));
	if (state->bdone) {
		// Skip block
		feeder->skip_until((c == '{') ? "}" : "\n");
	} else {
		if (c != '{')
			feeder->backup(1);
		
		feeder->set_end((c == '{') ? '}' : '\n');
		parse_global(feeder, context);

		// Reset terminal
		feeder->set_end();
	}

	// End the current branch
	state->branch = false;
	state->bdone = false;

	return true;
}

bool check_while(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	// Skip construction step or something
	node_manager ncond(ctx, paren);

	while (isspace(c = feeder->feed()));
	if (is_true(ncond.value(ctx))) {
		if (c != '{')
			feeder->backup(1);
		
		feeder->set_end((c == '{') ? '}' : '\n');

		Pardon pardon;
		node_manager nloop = cc_parse(feeder, ctx, {}, pardon);

		node_manager nwhile;

		nwhile.set_label(l_while_loop);
		nwhile.append(ncond);
		nwhile.append(nloop);

		nwhile.value(ctx);

		// Reset terminal
		feeder->set_end();
	} else {
		// Skip block
		feeder->skip_until((c == '{') ? "}" : "\n");
	}

	return true;
}

bool check_for(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	using namespace std;
	cout << "paren = " << paren << endl;

	// Skip construction step or something
	node_manager niter(ctx, paren);

	niter.print();

	node lin = niter.tree();

	std::cout << strlabs[lin.label()] << std::endl;

	if (lin.label() != l_generator_in)
		throw bad_for();
	
	lvalue *lv = lin[0].cast <lvalue> ();

	if (!lv)
		throw bad_for();
	
	std::string ident = lv->symbol();

	std::cout << "IDENT = " << ident << endl;

	Pardon pardon {ident};

	while (isspace(c = feeder->feed()));
	if (c != '{')
		feeder->backup(1);
		
	feeder->set_end((c == '{') ? '}' : '\n');

	node_manager nloop = cc_parse(feeder, ctx, {}, pardon);

	node_manager nfor;

	nfor.set_label(l_for_loop);
	nfor.append(niter);
	nfor.append(nloop);

	nfor.print();

	nfor.value(ctx);

	// Reset terminal
	feeder->set_end();

	return true;
}

void check_keyword(std::string &cache,
		Feeder *feeder,
		Engine *context,
		State *state)
{
	using Processor = std::function <bool (Feeder *, Engine *, State *)>;
	static const Symtab <Processor> keywords {
		{"if", check_if},
		{"elif", check_elif},
		{"else", check_else},
		{"while", check_while},
		{"for", check_for}
	};

	if (keywords.find(cache) == keywords.end())
		return;
	
	// Clear cache if the keyword extraction is successful
	if ((keywords.at(cache))(feeder, context, state))
		cache.clear();
}

}
