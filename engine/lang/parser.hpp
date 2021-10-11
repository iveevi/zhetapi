#ifndef PARSER_H_
#define PARSER_H_

// Standard headers
#include <functional>
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

// Symbol table class
class SymbolTable {
public:
	// Index type: (scope offset, index)
	using Index = std::pair <int, int>;
private:
	// Upper/previous level
	SymbolTable *		_next = nullptr;

	// Value array and index map
	std::vector <Variant>	_values;
	Strtable <size_t>	_indices;

	// TODO: print function for debugging

	// Private find function
	void _find(const std::string &str, Index &index) {
		Index ni = {-1, -1};
		if (_indices.find(str) != _indices.end()) {
			index.second = _indices[str];
		} else if (_next) {
			index.first++;
			_next->_find(str, index);
		} else {
			index = ni;
		}
	}
public:
	void link_up(SymbolTable *symtab) {
		_next = symtab;
	}

	Index find(const std::string &str) {
		Index index;
		_find(str, index);
		return index;
	}

	// Get values/variables
	Variant get(int soff, int index) const {
		if (soff > 0)
			return _next->get(soff - 1, index);
		else if (soff < 0)
			return nullptr;
		
		// TODO: need to check bounds
		return _values[index];
	}

	Variant get(const Index &index) const {
		return get(index.first, index.second);
	}

	Variant get(const std::string &str) {
		Index index = find(str);
		if (index.first != -1)
			return get(index);
		return nullptr;
	}

	// Set values/variables
	void set(Variant vptr, int soff, int index) {
		if (soff > 0)
			_next->set(vptr, soff - 1, index);
		else if (soff == 0) // TODO: need to check bounds
			_values[index] = vptr;

		// TODO: Warning here?
	}

	void set(Variant vptr, const Index &index) {
		return set(vptr, index.first, index.second);
	}

	void set(Variant vptr, const std::string &str) {
		Index index = find(str);
		if (index.first != -1)
			set(vptr, index.first, index.second);
	}
};

// Parser class
// TODO: add builtin functions, etc
class Parser {
	// _tsq for retrieving tokens
	// _store for storing these tokens
	//	so that we can restore them to
	//	the tsq later
	ads::TSQueue <void *> *	_tsq = nullptr;
	std::stack <void *>	_store;

	// Operation stack
	// TODO: change to Variant later
	std::stack <Primitive>	_stack;

	// Symbol table
	SymbolTable		_symtab;

	// Lexing variables
	LexTag			_bp;
public:
	// Public structs
	// TODO: add line number to this (and possibly char)
	struct TagPair {
		void *data;
		LexTag tag = get_ltag(data);
	};

	// Public aliases
	using VTags = std::vector <TagPair>;

	// Constructors
	Parser(ads::TSQueue <void *> *);

	~Parser();

	// Helper functions
	void set_bp(LexTag = DONE);
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
	bool statement();
	void algorithm();			// Return bool?

	// Ultimate function
	void run();

	// Debugging functions
	void dump();

	// Exceptions: TODO: need to add line numbers
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

	class bad_require : public std::runtime_error {
	public:
		bad_require(const std::string &str)
			: std::runtime_error("Parser: expected " + str) {}
	};
};

}

#endif
