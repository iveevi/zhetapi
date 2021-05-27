#ifndef PARSER_H_
#define PARSER_H_

// C/C++ headers
#include <iostream>
#include <exception>

// Engine headers
#include "feeder.hpp"
#include "../engine.hpp"

namespace zhetapi {

// Parsing state
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

// No forwarding needed
void run(const std::string &, Engine *);

void check_keyword(std::string &, Feeder *, Engine *, State *);

// Make part of public API so others and I can use
int parse_global(const std::string &, Engine *);
int parse_global(Feeder *, Engine *);

// Compile parsers and runners
node_manager cc_run(const std::string &,
		Engine *,
		const Args &,
		Pardon &);

node_manager cc_keyword(std::string &,
		Feeder *,
		Engine *,
		const Args &,
		Pardon &,
		State *);

node_manager cc_parse(Feeder *,
		Engine *,
		const Args &,
		Pardon &);

// Exceptions ()
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

}

#endif
