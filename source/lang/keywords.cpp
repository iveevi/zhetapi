#include "../../engine/lang/parser.hpp"
#include "../../engine/lang/error_handling.hpp"
#include "../../engine/core/node_manager.hpp"
#include <cstdlib>

namespace zhetapi {

// TODO: make sure to guard for EOFs

// TODO: put elsewhere
bool is_true(Token *tptr)
{
	static const Operand <bool> true_token(true);

	return true_token == tptr;
}

// TODO: put elsewhere (helpers)

// Importing DLL
static void import_dll(const std::string &file, Module &module, const char *linted_version)
{
	const char *dlsymerr = nullptr;

	// Load the library
	void *handle = dlopen(file.c_str(), RTLD_NOW);

	// Check for errors
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: unable to open file '%s': %s\n", file.c_str(), dlsymerr);

		abort();
	}
	
	// Display the linted version
	void *ptr1 = dlsym(handle, "__zhp_linted_version__");
	
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: could not find \"__zhp_linted_version__\" in file '%s': %s\n", file.c_str(), dlsymerr);

		abort();
	}

	const char *(*lver)() = (const char *(*)()) ptr1;

	if (!lver) {
		printf("Failed to extract version linter\n");

		abort();
	}

	// TODO: color in yellow
	if (strcmp(linted_version, lver()) != 0) {
		// TODO: add color macros
		printf("\x1b[1;33m" "WARNING:" "\x1b[0m" " Version of module \"%s\", <%s>, "
				"differs from interpreter's version, <%s>\n",
				file.c_str(), lver(), linted_version);
	}

	// Get the exporter
	void *ptr2 = dlsym(handle, "zhetapi_export_symbols");

	// Check for errors
	dlsymerr = dlerror();

	if (dlsymerr) {
		printf("Fatal error: could not find \"zhetapi_export_symbols\" in file '%s': %s\n", file.c_str(), dlsymerr);

		abort();
	}

	Module::Exporter exporter = (Module::Exporter) ptr2;

	if (!exporter) {
		printf("Failed to extract exporter\n");

		abort();
	}

	exporter(&module);
}

// Importing libraries
static void import_as(const std::string &lib, const std::string &alias, Engine *ctx, State *state)
{
	// First check the possible locations and warn on ambiguity
	std::string fname = lib.substr(0, lib.find_last_of("."));

	Args lpaths;
	bool dll = false;

	for (const std::string &idir : state->idirs) {
		// Generate library and file names as candidates
		std::string join1 = idir + '/' + lib + ".zhplib";
		std::string join2 = idir + '/' + lib + ".zhp";

		std::cout << "trying : \"" << join1 << "\"" << std::endl;
		std::cout << "trying : \"" << join2 << "\"" << std::endl;

		// Check dll: use dlopen to check presence
		void *handle = dlopen(join1.c_str(), RTLD_NOW);

		// std::fstream f1(join1);
		if (handle != nullptr) {
			lpaths.push_back(join1);
			dll = true;
	
			// close handle
			dlclose(handle);
		}

		std::fstream f2(join2);
		if (f2) {
			lpaths.push_back(join2);
			dll = false;
		}
	}

	// TODO: Fancify the errors! (and throw if necessary)
	if (lpaths.size() < 1) {
		std::cout << "Error: could not find library \"" << lib << "\"" << std::endl;

		// TODO: throw
		return;
	} else if (lpaths.size() > 1) {
		std::cout << "Ambiguity in importing \"" << lib << "\"" << std::endl;

		for (const std::string &str : lpaths)
			std::cout << "\tCandidate " << str << std::endl;

		// TODO: throw
		return;
	}

	// TODO: deal with lib

	// Create, read and load library (depending on file or lib)
	Module *module = new Module(alias);

	if (dll) {
		import_dll(lpaths[0], *module, state->lver);
	} else {
		StringFeeder sf = file_feeder(lpaths[0]);
		mdl_parse(&sf, ctx, module);
	}
	
	// TODO: remove name info from modules
	ctx->put(alias, module);
}

// TODO: make sure state->branch gets reset
static OpZ *check_if(Feeder *feeder,
		Engine *context,
		State *state)
{
	// Save end
	Feeder::State end = feeder->get_end();

	// Update the state
	state->branch = true;

	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	// Skip construction step or something
	node_manager ncond(context, paren);
	
	if (ncond.get_label() == l_generator_in) {
		ncond.set_label(l_set_in);

		lvalue *lv = ncond[0].cast <lvalue> ();

		// TODO: Throw an error that redirects to the issue page
		// if (!lv) throw

		// TODO: Gotta free or something
		ncond[0].retokenize(new rvalue(lv->symbol()));
	}
	
	Token *tptr = ncond.value(context);

	// TODO: Add a skip whitespace
	while (isspace(c = feeder->feed()));
	if (is_true(tptr)) {
		state->bdone = true;
		if (c != '{')
			feeder->backup(1);

		feeder->set_end((c == '{') ? '}' : '\n');
		parse_global(feeder, context, state->idirs);

		// Reset terminal
		feeder->set_end(end);
	} else {
		// TODO: add a skip until for characters
		feeder->skip_until((c == '{') ? "}" : "\n");
	}

	return nullptr;
}

static OpZ *check_elif(Feeder *feeder,
		Engine *context,
		State *state)
{
	// Save end
	Feeder::State end = feeder->get_end();

	if (!state->branch)
		throw bad_elif();

	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	node_manager ncond(context, paren);
	
	if (ncond.get_label() == l_generator_in) {
		ncond.set_label(l_set_in);

		lvalue *lv = ncond[0].cast <lvalue> ();

		// TODO: Throw an error that redirects to the issue page
		// if (!lv) throw

		// TODO: Gotta free or something
		ncond[0].retokenize(new rvalue(lv->symbol()));
	}

	Token *tptr = ncond.value(context);

	while (isspace(c = feeder->feed()));
	if (!state->bdone && is_true(tptr)) {
		state->bdone = true;
		if (c != '{')
			feeder->backup(1);

		feeder->set_end({(c == '{') ? '}' : '\n', 1});
		parse_global(feeder, context, state->idirs);

		// Reset terminal
		feeder->set_end(end);
	} else  {
		// TODO: add a skip until for characters
		feeder->skip_until((c == '{') ? "}" : "\n");
	}

	return nullptr;
}

static OpZ *check_else(Feeder *feeder,
		Engine *context,
		State *state)
{
	// Save end
	Feeder::State end = feeder->get_end();

	// Throw exception
	if (!state->branch)
		throw bad_else();

	char c;

	while (isspace(c = feeder->feed()));
	if (state->bdone) {
		// Skip block
		feeder->skip_until((c == '{') ? "}" : "\n");
	} else {
		if (c != '{')
			feeder->backup(1);

		feeder->set_end((c == '{') ? '}' : '\n');
		parse_global(feeder, context, state->idirs);

		// Reset terminal
		feeder->set_end(end);
	}

	// End the current branch
	state->branch = false;
	state->bdone = false;

	return nullptr;
}

static OpZ *check_while(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	// Save end
	Feeder::State end = feeder->get_end();

	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();
	// cout << "paren = " << paren << endl;

	// Skip construction step or something
	node_manager ncond(ctx, paren);

	// ncond.print();

	while (isspace(c = feeder->feed()));
	if (is_true(ncond.value(ctx))) {
		if (c != '{')
			feeder->backup(1);

		feeder->set_end({(c == '{') ? '}' : '\n', 1});

		Pardon pardon;
		node_manager nloop = cc_parse(feeder, ctx, {}, pardon);

		node_manager nwhile;

		nwhile.set_label(l_while_loop);
		nwhile.append(ncond);
		nwhile.append(nloop);

		nwhile.value(ctx);

		// Restore terminal
		feeder->set_end(end);

		// cout << "Next character after nloop: " << (int) feeder->peek() << endl;
	} else {
		// Skip block
		feeder->skip_until((c == '{') ? "}" : "\n");
	}

	return nullptr;
}

static OpZ *check_for(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	// Save end
	Feeder::State end = feeder->get_end();

	char c;
	while ((c = feeder->feed()) != '(');

	std::string paren = feeder->extract_parenthesized();

	// Skip construction step or something
	node_manager niter(ctx, paren);

	node lin = niter.tree();
	if (lin.label() != l_generator_in)
		throw bad_for();

	lvalue *lv = lin[0].cast <lvalue> ();

	if (!lv)
		throw bad_for();

	std::string ident = lv->symbol();

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

	nfor.value(ctx);

	// Reset terminal
	feeder->set_end(end);

	return nullptr;
}

static OpZ *check_break(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	throw global_break();

	return nullptr;
}

static OpZ *check_continue(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	throw global_continue();

	return nullptr;
}

static OpZ *check_alg(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	// Save end
	Feeder::State end = feeder->get_end();

	std::pair <std::string, Args> sig = feeder->extract_signature();

	char c;
	while (isspace(c = feeder->feed()) && c != EOF);
	if (c != '{')
		feeder->backup(1);

	feeder->set_end({(c == '{') ? '}' : '\n', 1});

	Pardon pardon;
	node_manager nbody = cc_parse(feeder, ctx, sig.second, pardon);

	nbody.add_args(sig.second);
	nbody.set_label(l_sequential);
	nbody.compress_branches();

	algorithm alg(sig.first, "", sig.second, nbody);

	ctx->put(alg);

	// Reset terminal
	feeder->set_end(end);

	return nullptr;
}

static OpZ *check_return(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	std::string expression;

	// TODO: Allow multiline if the user add '\'
	char c;
	while ((c = feeder->feed()) != '\n' && c != EOF)
		expression += c;

	Token *tptr = node_manager(ctx, expression).value(ctx);

	OpZ *opz = dynamic_cast <OpZ *> (tptr);

	if (opz)
		return opz;

	throw global_int_return();

	return nullptr;
}

// include [dir1], [dir2], ...
static OpZ *check_include(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	// TODO: Allow multiline
	char c;

	std::string line;
	while ((c = feeder->feed()) != '\n' && c != EOF)
		line += c;

	Args vcomma = comma_split(line);
	for (auto str : vcomma)
		state->idirs.push_back(str);

	return nullptr;
}

static OpZ *check_import(Feeder *feeder,
		Engine *ctx,
		State *state)
{
	// TODO: Allow multiline
	// TODO: also allow 'as' clause
	char c;

	// TODO: Combine all here instead of using all the helpers (and reparsing...)
	std::string line;
	while ((c = feeder->feed()) != '\n' && c != EOF)
		line += c;

	Args vcomma = comma_split(line, false);
	for (auto str : vcomma) {
		std::cout << "Importing as: \"" << str << "\"" << std::endl;

		// TODO: alias please
		std::pair <std::string, std::string> pstr = as_split(str);

		std::cout << "\tpstr = <\""  << pstr.first << "\", \"" << pstr.second << "\">" << std::endl;

		import_as(pstr.first, pstr.second, ctx, state);
	}

	return nullptr;
}

static OpZ *check_global(Feeder *feeder,
		Engine *ctx,
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

		// TODO: should it just be run_assignment?
		if (veq.size() > 1)
			run_assignment(veq, ctx);
	}

	return nullptr;
}

OpZ *check_keyword(std::string &cache,
		Feeder *feeder,
		Engine *context,
		State *state)
{
	using Processor = std::function <OpZ *(Feeder *, Engine *, State *)>;
	static const Symtab <Processor> keywords {
		{"if", check_if},
		{"elif", check_elif},
		{"else", check_else},
		{"while", check_while},
		{"for", check_for},
		{"break", check_break},
		{"continue", check_continue},
		{"alg", check_alg},
		{"return", check_return},
		{"include", check_include},
		{"import", check_import},
		{"global", check_global}
	};

	if (keywords.find(cache) == keywords.end())
		return nullptr;

	// Execute and clear the cache
	OpZ *ret = (keywords.at(cache))(feeder, context, state);
	cache.clear();

	return ret;
}

}
