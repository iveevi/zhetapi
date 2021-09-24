#ifndef PARSER_H_
#define PARSER_H_

// Standard headers
#include <stack>
#include <stdexcept>

// Engine headers
#include "ltag.hpp"
#include "lexer.hpp"
#include "../ads/tsqueue.hpp"
#include "../core/common.hpp"
#include "../core/variant.hpp"

namespace zhetapi {

// Helper functions for Parser
inline bool is_operation(LexTag ltag)
{
	switch (ltag) {
	case PLUS: case MINUS:
	case TIMES: case DIVIDE:
		return true;
	default:
		return false;
	}

	return false;
}

// Checks equality is both combinations
inline bool excheq(LexTag ltag1, LexTag ltag2,
		LexTag eq1, LexTag eq2)
{
	return (ltag1 == eq1 && ltag2 == eq2)
		|| (ltag1 == eq2 && ltag2 == eq1);
}

// True if greater than or equal
inline bool opcmp(LexTag ltag1, LexTag ltag2)
{
	if (excheq(ltag1, ltag2, PLUS, MINUS))
		return true;
	if (excheq(ltag1, ltag2, TIMES, DIVIDE))
		return true;

	return (ltag1 >= ltag2);
}

// Parser class
// TODO: should only take a tsqueue of tags,
// for parallelization
class Parser {
	// _tsq for retrieving tokens
	// _store for storing these tokens
	//	so that we can restore them to
	//	the tsq later
	ads::TSQueue <void *> *	_tsq = nullptr;
	std::stack <void *>	_store;

	// Symbol table
	Strtable <Variant> symtab;
public:
	// Public structs
	struct TagPair {
		void *data;
		LexTag tag = get_ltag(data);
	};

	// Public aliases
	using VTags = std::vector <TagPair>;

	Parser(ads::TSQueue <void *> *);

	~Parser();

	// Helper functions
	TagPair get();
	TagPair require(LexTag);

	void backup();
	void backup(size_t);

	bool try_grammar(VTags &, const std::vector <LexTag> &);
	// TODO: need a require grammar function as well

	std::queue <TagPair> shunting_yard();

	// Grammatical functions
	Variant expression_imm();		// Private
	void function();
	void statement();
	void algorithm();			// Return bool?

	// Ultimate function
	void run();

	// Debugging functions
	void dump();

	// Exceptions
	class eoq : public std::runtime_error {
	public:
		eoq() : std::runtime_error("Parser: end of tag queue") {}
	};

	class bad_tag : public std::runtime_error {
	public:
		bad_tag(LexTag got, LexTag exp)
			: std::runtime_error("Parser: unexpected tag <"
				+ strlex[got] + ">, expected <"
				+ strlex[exp] + ">") {}
	};
};

}

#endif
