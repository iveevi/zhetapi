#ifndef PARSER_H_
#define PARSER_H_

// C/C++ headers
#include <iostream>
#include <exception>
#include <stdexcept>

// Engine headers
#include "feeder.hpp"
#include "../engine.hpp"
#include "../module.hpp"

namespace zhetapi {

// Parsing state (put in another file)
struct State {
	std::string	cached;

	size_t		line		= 0;

	// Nesting states
	size_t		parenthesis	= 0;
	size_t		braces		= 0;
	size_t		brackets	= 0;

	bool		branch		= false;
	bool		bdone		= false;

	bool		alg		= false;

	Args		idirs;

	const char *	lver;

	State(bool = false);

	void check_nest(char);
	bool is_nested();
};

// Helper functions
bool is_vaild_ident_char(char, size_t);
bool is_valid_ident(const std::string &);
bool in_args(const Args &, const Args &);
Args get_args(const std::string &);
Args eq_split(const std::string &);
Args comma_split(const std::string &);

// No forwarding needed
void run(const std::string &, Engine *, bool = false);
void run_normal(const std::string &, Engine *, bool = false);
void run_assignment(const Args &, Engine *, bool = false);

OpZ *check_keyword(std::string &, Feeder *, Engine *, State *);

// Make part of public API so others and I can use
int parse_global(Feeder *, Engine *, const Args & = {".", "/usr/local/include/zhp"}, const char * = "Untitled", bool = false);

// Make part of public API so others and I can use
void mdl_parse(Feeder *, Engine *, Module *);

// Compile parsers and runners
node_manager cc_run_assignment(
		const Args &,
		Engine *,
		const Args &,
		Pardon &);

node_manager cc_run(
		const std::string &,
		Engine *,
		const Args &,
		Pardon &);

node_manager cc_keyword(
		std::string &,
		Feeder *,
		Engine *,
		const Args &,
		Pardon &,
		State *);

node_manager cc_parse(
		Feeder *,
		Engine *,
		const Args &,
		Pardon &);

// TODO: Exceptions (put into another header)
// It is to be noted that branching does not count as nesting
class bad_identifier : public std::runtime_error {
public:
	bad_identifier(const std::string &str)
		: std::runtime_error(str) {}
};

class bad_line : public std::runtime_error {
public:
	bad_line(const std::string &str)
		: std::runtime_error(str) {}
};

class args_mismatch : public std::runtime_error {
public:
	args_mismatch(const std::string &str)
		: std::runtime_error(str) {}
};

class bad_elif : public std::runtime_error {
public:
	bad_elif()
		: std::runtime_error("Cannot have elif without if before") {}
};

class bad_else : public std::runtime_error {
public:
	bad_else()
		: std::runtime_error("Cannot have else without if before") {}
};

class bad_for : public std::runtime_error {
public:
	bad_for()
		: std::runtime_error("Expected ([identifier] in [expression]) inside the clause of for-loop") {}
};

class global_break : public std::runtime_error {
public:
	global_break()
		: std::runtime_error("Cannot break in global scope") {}
};

class global_continue : public std::runtime_error {
public:
	global_continue()
		: std::runtime_error("Cannot use continue in global scope") {}
};

class nested_alg : public std::runtime_error {
public:
	nested_alg()
		: std::runtime_error("Cannot define an algorithm in a nested scope") {}
};

class global_int_return : public std::runtime_error {
public:
	global_int_return()
		: std::runtime_error("Can only return integer codes at global scope") {}
};

class nested_include : public std::runtime_error {
public:
	nested_include()
		: std::runtime_error("Cannot include directories from a nested scope") {}
};

class nested_import : public std::runtime_error {
public:
	nested_import()
		: std::runtime_error("Cannot import from a nested scope") {}
};

class nested_global : public std::runtime_error {
public:
	nested_global()
		: std::runtime_error("Cannot use \"global\" in a nested scope") {}
};

class fatal_error : public std::runtime_error {
public:
	fatal_error(const char *file, const char *function)
		: std::runtime_error("Fatal error: at " + std::string(file)
				+ " : " + std::string(function)
				+ "\n\t\tPlease report this to "
				"https://github.com/vedavamadathil/zhetapi/issues") {}
};

}

#endif
