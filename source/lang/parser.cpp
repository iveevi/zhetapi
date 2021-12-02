#include "../../engine/lang/parser.hpp"

// Standard headers
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Engine headers
#include "../../engine/core/genoptns.hpp"
#include "../../engine/lang/iseq.hpp"

namespace zhetapi {

// Parser class
Parser::Parser(ads::TSQueue <void *> *queue) : _tsq(queue) {}

// TODO: at the end, deallocate only from both _store
// the _tsq will be freed outside of this parser
Parser::~Parser() {}

// Getter helper
void Parser::set_bp(LexTag ltag)
{
	_bp = ltag;
}

Parser::TagPair Parser::get()
{
	if (_tsq->empty())
		throw eoq();

	void *ptr = _tsq->pop();
	_store.push(ptr);

	if (get_ltag(ptr) == _bp) {
		std::cout << "HIT BREAKPOINT TOKEN" << std::endl;

		// TODO: need to allocate this in extra vector
		return {new NormalTag {DONE::id}};
	}

	return {ptr};
}

// Requirement function
Parser::TagPair Parser::require(LexTag ltag)
{
	auto pr = get();
	std::cout << "REQ-GOT Tag: " << strlex[pr.tag] << std::endl;

	if (pr.tag != ltag)
		throw bad_tag(pr.tag, ltag);

	return pr;
}

// Backup transfer from _store to _tsq
void Parser::backup()
{
	void *ptr = _store.top();
	// std::cout << "BACKUP: " << str_ltag(ptr) << std::endl;
	_store.pop();
	_tsq->push_front(ptr);
}

void Parser::backup(size_t n)
{
	for (size_t i = 0; i < n; i++)
		backup();
}

// Specialized grammars
template <>
Variant Parser::grammar <PRIMITIVE::id> ()
{
	auto pr = get();

	// TODO: should we do this once at the start of
	// the grammar or allow whitespace between tokens?
	// TODO: use a function is_whitespace
	while (pr.tag == NEWLINE::id) pr = get();

	if (pr.tag == PRIMITIVE::id)
		return pr.data;

	backup();
	return nullptr;
}

/*
template <>
Variant Parser::grammar <gr_operand> ()
{
	Variant vt = nullptr;
	if ((vt = do_grammar <PRIMITIVE> ()))
		return vt;
	if ((vt = do_grammar <STRING> ()))
		return vt;

	return nullptr;
}

template <>
Variant Parser::grammar <gr_closed_factor> ()
{
	// Multigrammars
	multigrammar <LPAREN, gr_expression, RPAREN> paren(this);

	Variant vt = nullptr;
	if ((vt = do_grammar <gr_operand> ()))
		return vt;

	Values values;
	if (paren(values))
		return values[0];

	return nullptr;
}

template <>
Variant Parser::grammar <gr_full_factor> ()
{
	Variant vt = nullptr;

	// Last case
	if ((vt = do_grammar <gr_closed_factor> ()))
		return vt;

	return nullptr;
}

template <>
Variant Parser::grammar <gr_factor> ()
{
	Variant vt = nullptr;

	/* Juxtaposition of factors as multiplication
	// std::cout << "\nFACTOR (1st branch)" << std::string(50, '=') << std::endl;
	Values values;
	multigrammar <gr_full_factor, gr_factor> mg(this);
	if (mg(values)) {
	//	std::cout << "Size of values = " << values.size() << std::endl;
	//	std::cout << "Juxtaposition of multiplication: need to implement" << std::endl;
		throw int(1);
	} *

	// Last case
	if ((vt = do_grammar <gr_full_factor> ()))
		return vt;

	return nullptr;
}

template <>
Variant Parser::grammar <gr_term> ()
{
	Variant vt = nullptr;

	// Last case
	if ((vt = do_grammar <gr_factor> ()))
		return vt;

	return nullptr;
}

template <>
Variant Parser::grammar <gr_simple_expression> ()
{
	multigrammar <gr_term, PLUS, gr_term> plus_gr(this);
	multigrammar <gr_term, MINUS, gr_term> minus_gr(this);

	Values values;
	if (plus_gr(values)) {
		return core::compute(core::l_add,
			vt_cast(values[0]),
			vt_cast(values[2])
		);
	} else if (minus_gr(values)) {
		return core::compute(core::l_sub,
			vt_cast(values[0]),
			vt_cast(values[2])
		);
	}

	return nullptr;
}

template <>
Variant Parser::grammar <gr_expression> ()
{
	Variant vt;
	if ((vt = do_grammar <gr_simple_expression> ()))
		return vt;

	return nullptr;
} */

// Debugging
void SymbolTable::dump() const
{
	// TODO: add a size function
	std::cout << std::string(50, '-') << std::endl;
	std::cout << "Symbol table dump:" << std::endl;
	std::cout << "Size: " << _indices.size() << std::endl;
	std::cout << std::string(50, '-') << std::endl;

	size_t i = 0;
	for (const auto &pr : _indices) {
		std::cout << pr.first << "\t" << pr.second
			<< "\t|\t" << i << "\t" << variant_str(_values[i++])
			<< std::endl;
	}
}

void Parser::dump()
{
	_symtab.dump();
}

}
