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

// Colors
#define RESET        "\033[0m"
#define RED          "\033[31;1m"
#define GREEN        "\033[32;1m"
#define YELLOW       "\033[33;1m"

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
		if (index.first != -1) {
			set(vptr, index.first, index.second);
		} else {
			_indices[str] = _values.size();
			_values.push_back(vptr);
		}
	}

	// Print function
	void dump() const;
};

#define ZHP_PARSER_DEBUG

// Parser class
// TODO: add builtin functions, etc
class Parser {
public:
	// Public structs
	// TODO: add line number to this (and possibly char)
	struct TagPair {
		void *data;
		LexTag tag = get_ltag(data);
	};

	// Public aliases
	using VTags = std::vector <TagPair>;
	using Values = std::vector <Variant>;
private:
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

	// Multigrammar private structs
	template <LexTag ... Codes>
	struct _multigrammar {
		_multigrammar(Parser *) {}

		bool operator()(Values &) {
			return true;
		}
	};

	template <LexTag code, LexTag ... Codes>
	class _multigrammar <code, Codes...> {
		Parser *_parser;
	public:
		_multigrammar(Parser *parser)
				: _parser(parser) {}

		bool operator()(Values &values) {
			// std::cout << "_multigrammar @ " << strlex[code] << std::endl;

			// Check and add
			Variant vt;

			_multigrammar <Codes...> next(_parser);
			if ((vt = _parser->do_grammar <code> ())) {
			//	std::cout << "\tGOT VALUE vt = " << vt << std::endl;
				values.push_back(vt);
				return next(values);
			} /* else {
				_parser->backup();
			} */

			return false;
		}
	};

#ifdef ZHP_PARSER_DEBUG

	size_t			_indents = 0;

#endif

public:
	// Constructors
	Parser(ads::TSQueue <void *> *);

	~Parser();

	// Helper functions
	void set_bp(LexTag = DONE);
	TagPair get();
	TagPair require(LexTag);

	void backup();
	void backup(size_t);

	// Generic grammar function:
	// 	default specialization checks if
	// 	the next tag has the specified code
	//
	// 	specializations are used to build
	// 	up the grammar
	template <LexTag code>
	Variant grammar() {
		auto pr = get();

		// TODO: should we do this once at the start of
		// the grammar or allow whitespace between tokens?
		// TODO: use a function is_whitespace
		while (pr.tag == NEWLINE) pr = get();

		// std::cout << "@GRAMMAR <" << strlex[code] << "> actually is " << strlex[pr.tag] << std::endl;
		if (pr.tag == code)
			return new NormalTag {code};

		backup();
		return nullptr;
	}

	// Debug wrapped of grammar:
	// 	use while debugging to
	// 	print tracebacks of parsing
	template <LexTag code>
	Variant do_grammar(Variant fail = nullptr) {

#ifdef ZHP_PARSER_DEBUG

		static const int spacing = 4;

		// Store returns here
		Variant vt;

		std::cout << std::string((spacing * _indents++), ' ')
			<< "<" << strlex[code] << ">" << std::endl;

		vt = grammar <code> ();

		std::string indent(spacing * _indents, ' ');
		if (vt != fail) {
			std::cout << indent << GREEN << "SUCCESS: "
				<< RESET << vt << std::endl;
		} else {
			std::cout << indent << RED << "FAIL"
				<< RESET << std::endl;
		}

		std::cout << std::string((spacing * (--_indents)), ' ')
			<< "</" << strlex[code] << ">" << std::endl;

		return vt;

#else

		return grammar <code> ();

#endif

	}

	// Multigrammar struct
	template <LexTag ... Codes>
	struct multigrammar {
		Parser *_parser;
	public:
		multigrammar(Parser *parser)
				: _parser(parser) {}

#ifdef ZHP_PARSER_DEBUG

		bool kernel(Values &values) {
			_multigrammar <Codes...> mg(_parser);

			if (mg(values))
				return true;

			// Need to empty values if not successful
			values.clear();
			return false;
		};

		bool operator()(Values &values) {
			static const int spacing = 4;

			// Store returns here
			bool bl;

			std::cout << std::string((spacing * _parser->_indents++), ' ')
				<< "<multigrammar>" << std::endl;

			bl = kernel(values);

			std::string indent(spacing * _parser->_indents, ' ');
			if (bl) {
				std::cout << indent << GREEN << "SUCCESS"
					<< RESET << std::endl;
			} else {
				std::cout << indent << RED << "FAIL"
					<< RESET << std::endl;
			}

			std::cout << std::string((spacing * (--_parser->_indents)), ' ')
				<< "</multigrammar>" << std::endl;

			return bl;

		}

#else

		bool operator()(Values &values) {
			_multigrammar <Codes...> mg(_parser);

			if (mg(values))
				return true;

			// Need to empty values if not successful
			values.clear();
			return false;
		};
#endif

	};

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

// Grammar specializations
template <>
Variant Parser::grammar <PRIMITIVE> ();

template <>
Variant Parser::grammar <gr_operand> ();

template <>
Variant Parser::grammar <gr_closed_factor> ();

template <>
Variant Parser::grammar <gr_full_factor> ();

template <>
Variant Parser::grammar <gr_factor> ();

template <>
Variant Parser::grammar <gr_term> ();

template <>
Variant Parser::grammar <gr_simple_expression> ();

template <>
Variant Parser::grammar <gr_expression> ();

}

#endif
