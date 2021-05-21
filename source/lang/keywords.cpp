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
		Engine *context,
		State *state)
{
	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	// Skip construction step or something
	node_manager nm(context, paren);

	nm.print();

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
		{"while", check_while}
	};

	if (keywords.find(cache) == keywords.end())
		return;
	
	// Clear cache if the keyword extraction is successful
	if ((keywords.at(cache))(feeder, context, state))
		cache.clear();
}

}
