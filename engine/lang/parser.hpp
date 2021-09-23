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

namespace zhetapi {

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

	/* Symbol table: string to index
	Strtable <size_t>	_hash;

	// Symbol table: index to value
	std::vector <Variant>	_vregs; */

	// Private structs
	struct TagPair {
		void *data;
		LexTag tag;
	};

	// Private aliases
	using VTags = std::vector <TagPair>;
public:
	Parser(ads::TSQueue <void *> *);

	~Parser();

	// Helper functions
	TagPair get();
	TagPair require(LexTag);

	void backup();
	void backup(size_t);

	bool try_grammar(VTags &, const std::vector <LexTag> &);

	// Grammatical functions
	void expression_imm();		// Private
	void statement();
	void algorithm();

	// Ultimate function
	void run();

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
