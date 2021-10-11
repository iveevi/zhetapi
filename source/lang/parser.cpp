#include "../../engine/lang/parser.hpp"

// Standard headers
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Engine headers
#include "../../engine/core/primoptns.hpp"
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
		return {new NormalTag {DONE}};
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
	_store.pop();
	_tsq->push_front(ptr);
}

void Parser::backup(size_t n)
{
	for (size_t i = 0; i < n; i++)
		backup();
}

// Try grammar function
bool Parser::try_grammar(VTags &tags, const std::vector <LexTag> &codes)
{
	for (LexTag c : codes) {
		// Get the next non-newline tag
		auto pr = get();
		while (pr.tag == NEWLINE)
			pr = get();

		std::cout << "try-tag -> " << strlex[pr.tag] << std::endl;
		if (pr.tag != c) {
			// Backup all the tags read in the attempt
			//	plus one for the current one
			backup(tags.size() + 1);
			tags.clear();
			return false;
		}
		tags.push_back(pr);
	}

	return true;
}

std::queue <Parser::TagPair> Parser::shunting_yard()
{
	// Shunting yard algorithm (stack)
	std::stack <TagPair> stack;
	std::queue <TagPair> queue;		// TODO: should this be tsqueue?

	// std::cout << "Immediate expression" << std::endl;
	while (true) {
		TagPair pr = get();
		// std::cout << pr.tag << std::endl;

		// First check for the newline
		if (pr.tag == NEWLINE) {
			// std::cout << "\tEnd of expression" << std::endl;
			break;
		}

		if (is_operation(pr.tag)) {
			while (!stack.empty()) {
				auto pr2 = stack.top();

				if (opcmp(pr2.tag, pr.tag)) {
					stack.pop();
					queue.push(pr2);
				} else {
					break;
				}
			}

			stack.push(pr);
		} else {
			queue.push(pr);
		}
	}

	// Transfer the remaining tags
	while (!stack.empty()) {
		queue.push(stack.top());
		stack.pop();
	}

	/* Show all the tokens in the queue
	std::cout << "Resulting queue..." << std::endl;
	while (!queue.empty()) {
		auto pr = queue.front();
		queue.pop();

		std::cout << "\ttag -> " << to_string(pr.data) << std::endl;
	} */

	return queue;
}

Variant Parser::expression_imm()
{
	// Shunting yard algorithm (stack)
	std::stack <TagPair> stack;
	std::queue <TagPair> queue;		// TODO: should this be tsqueue?

	std::cout << "Immediate expression" << std::endl;
	while (true) {
		TagPair pr = get();

		// First check for the newline
		if (pr.tag == NEWLINE || pr.tag == DONE) {
			// std::cout << "\tEnd of expression" << std::endl;

			// TODO: backup
			break;
		}

		if (is_operation(pr.tag)) {
			while (!stack.empty()) {
				auto pr2 = stack.top();

				if (opcmp(pr2.tag, pr.tag)) {
					stack.pop();
					queue.push(pr2);
				} else {
					break;
				}
			}

			stack.push(pr);
		} else {
			std::cout << "must be operand: " << strlex[pr.tag] << " = "
				<< PrimitiveTag::cast(pr.data).str() << std::endl;
			queue.push(pr);
		}
	}

	// Transfer the remaining tags
	while (!stack.empty()) {
		queue.push(stack.top());
		stack.pop();
	}

	// Show all the tokens in the queue
	/* std::cout << "Resulting queue..." << std::endl;
	while (!queue.empty()) {
		auto pr = queue.front();
		queue.pop();

		std::cout << "\ttag -> " << to_string(pr.data) << std::endl;
	} */

	// TODO: phase 2, computation
	// TODO: combine these two phases

	// TODO: stack should be variants
	std::stack <Primitive> operands;

	while (!queue.empty()) {
		TagPair pr = queue.front();
		queue.pop();

		// TODO: need to check if operands has enough operands!!
		if (is_operation(pr.tag)) {
			// TODO: need to check for number of operations
			OpCode code = OpCode(pr.tag - PLUS);

			// TODO: should be for variants instead
			// TODO: use functions (lambda) for popping

			// Note that the second argument comes first
			Primitive arg2 = operands.top();
			operands.pop();

			Primitive arg1 = operands.top();
			operands.pop();
			std::cout << "OP (" << strlex[pr.tag] << ") A1=" << arg1.str()
				<< ", A2=" << arg2.str() << std::endl;

			Primitive out = do_prim_optn(code, arg1, arg2);

			std::cout << "\tOUT=" << out.str() << std::endl;

			// Push output back onto stack
			operands.push(out);
		} else {
			operands.push(PrimitiveTag::cast(pr.data));
		}
	}

	// TODO: need to make sure there is only one element in the stack
	Primitive *out = new Primitive(operands.top());
	return Variant(out);
}

// Parse a function
void Parser::function()
{
	// skip newline for now
	require(NEWLINE);

	// Parse argument pack
	// TODO: new function, it is reused elsewhere
	std::string ident;
	Args args;

	// TODO: need a get function that checks for empty-ness
	TagPair pr = require(IDENTIFIER);
	ident = IdentifierTag::cast(pr.data);

	require(LPAREN);
	do {
		pr = get();
		if (pr.tag == IDENTIFIER) {
			args.push_back(IdentifierTag::cast(pr.data));

			pr = get();
			if (pr.tag != COMMA && pr.tag != RPAREN) {
				throw std::runtime_error(
					"Parser: expected <COMMA> or"
					" <RPAREN> after parameter"
				);
			}
		}
	} while (pr.tag != RPAREN);

	/* Post analysis
	std::cout << "FINAL: [" << ident << "][";
	for (size_t i = 0; i < args.size(); i++) {
		std::cout << args[i];
		if (i < args.size() - 1)
			std::cout << ", ";
	}
	std::cout << "]" << std::endl; */

	// Require equals after, etc
	require(ASSIGN_EQ);

	// Convert the remaining expression into postfix
	std::queue <TagPair> postfix = shunting_yard();

	// Store the function in the symbol table
	ISeq *iseq = new ISeq(postfix, args);
	iseq->dump();
	Variant vt = (Variant) new Object(mk_iseq(iseq));
	// _symtab[ident] = vt;
	_symtab.set(vt, ident);
}

// Parse statement
bool Parser::statement()
{
	VTags vtags;

	if (try_grammar(vtags, {IDENTIFIER, ASSIGN_EQ})) {
		std::string ident = IdentifierTag::cast(vtags[0].data);
		std::cout << "\tlooking for an expression now..." << std::endl;
		std::cout << "\tident was " << ident << std::endl;
		Variant vt = expression_imm();
		// _symtab[ident] = vt;
		_symtab.set(vt, ident);
		std::cout << "RESULT=" << variant_str(vt) << std::endl;
		return true;
	} else if (try_grammar(vtags, {IDENTIFIER, LPAREN})) {
		std::cout << "Function call..." << std::endl;
		auto pr = get();

		std::string ident = IdentifierTag::cast(vtags[0].data);
		std::cout << "Ident: " << ident << std::endl;
		if (pr.tag == RPAREN) {
			std::cout << "Empty function call..." << std::endl;
		} else {
			backup(1);
			set_bp(RPAREN);
			
			std::cout << "Expression: " << std::endl;
			Variant vt = expression_imm();
			std::cout << "RESULT=" << variant_str(vt) << std::endl;
		}

		// Reset break point
		set_bp();
	} else if (expression_imm()) {
		std::cout << "Expression as last resort succeeded" << std::endl;
	}

	return false;
}

// Parse algorithms
void Parser::algorithm()
{
	std::string ident;
	Args args;

	// TODO: need a get function that checks for empty-ness
	TagPair pr = require(IDENTIFIER);
	ident = IdentifierTag::cast(pr.data);

	require(LPAREN);

	// Loop through all available parameters
	do {
		pr = get();
		if (pr.tag == IDENTIFIER) {
			args.push_back(IdentifierTag::cast(pr.data));

			pr = get();
			if (pr.tag != COMMA && pr.tag != RPAREN) {
				throw std::runtime_error(
					"Parser: expected <COMMA> or"
					" <RPAREN> after parameter"
				);
			}
		}
	} while (pr.tag != RPAREN);

	// Post analysis
	std::cout << "FINAL: [" << ident << "][";
	for (size_t i = 0; i < args.size(); i++) {
		std::cout << args[i];
		if (i < args.size() - 1)
			std::cout << ", ";
	}
	std::cout << "]" << std::endl;
}

// Final run function
void Parser::run()
{
	while (!_tsq->empty()) {
		auto pr = get();

		std::cout << "tag = " << strlex[pr.tag] << std::endl;

		if (pr.tag == ALGORITHM) {
			std::cout << "ALGORITHM!!" << std::endl;
			algorithm();
		} else {
			// The line must be a statement
			backup(1);
			if (!statement())
				throw bad_require("statement");
		}
	}

	std::cout << "Done." << std::endl;
}

// Debugging
void Parser::dump()
{
	// Dashes
	std::string dash1;
	std::string dash2;
	std::string dash3;

	// Creating the dashes
	for (size_t i = 0; i < 55; i++) dash1 += "\u2500";
	for (size_t i = 0; i < 27; i++) dash2 += "\u2500";
	for (size_t i = 0; i < 27; i++) dash3 += "\u2500";

	std::cout << "\u250C" << dash1
		<< "\u2510" << std::endl;
	std::cout << "\u2502 " << std::left
		<< std::setw(53) << "Parser Dump"
		<< " \u2502" << std::endl;
	std::cout << "\u251C" << dash2 << "\u252C"
		<< dash3 << "\u2524" << std::endl;
	std::cout << "\u2502 "
		<< std::setw(25) << "Variable"
		<< " \u2502 " << std::setw(25) << "Value"
		<< " \u2502" << std::endl;
	std::cout << "\u251C" << dash2 << "\u253C"
		<< dash3 << "\u2524" << std::endl;
	/* for (const auto &pr : _symtab) {
		std::cout << "\u2502 " << std::setw(25)
			<< pr.first << " \u2502 "
			<< std::setw(25) << variant_str(pr.second)
			<< " \u2502 " << std::endl;
	} */
	// TODO: should go into a print function for symtab
	std::cout << "\u2514" << dash2 << "\u2534"
		<< dash3 << "\u2518" << std::endl;
}

}
