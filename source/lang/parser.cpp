#include "../../engine/lang/parser.hpp"

// Standard headers
#include <iostream>

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
Parser::TagPair Parser::get()
{
	if (_tsq->empty())
		throw eoq();

	void *ptr = _tsq->pop();
	_store.push(ptr);
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

		std::cout << "try-pr -> " << strlex[pr.tag] << std::endl;
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

			Primitive out = do_prim_optn(code, arg1, arg2);

			std::cout << "OP (" << strlex[pr.tag] << ") A1=" << arg1.str()
				<< ", A2=" << arg2.str() << ", OUT=" << out.str() << std::endl;

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
	symtab[ident] = vt;
}

// Parse statement
void Parser::statement()
{
	VTags vt;

	if (try_grammar(vt, {IDENTIFIER, ASSIGN_EQ})) {
		std::cout << "\tlooking for an expression now..." << std::endl;
		std::cout << "\tident was " << IdentifierTag::cast(vt[0].data) << std::endl;
		Variant vt = expression_imm();
		std::cout << "RESULT=" << variant_str(vt) << std::endl;
	}
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

		std::cout << "Tag = " << strlex[pr.tag] << std::endl;

		if (pr.tag == ALGORITHM) {
			std::cout << "ALGORITHM!!" << std::endl;
			algorithm();
		} /* else if (pr.tag == IDENTIFIER) {
			cout << "STATEMENT!!" << endl;
			// statement
			backup();
			statement();
		} */
	}

	std::cout << "Done." << std::endl;
}

// Debugging
void Parser::dump()
{
	std::cout << "Parser Dump:" << std::endl;
	for (const auto &pr : symtab) {
		std::cout << pr.first << "\t"
			<< variant_str(pr.second) << std::endl;
	}
}

}
