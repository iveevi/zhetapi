#include "../../engine/lang/parser.hpp"

namespace zhetapi {

Parser::Parser(ads::TSQueue <void *> *queue) : _tsq(queue) {}

// TODO: at the end, deallocate only from both _store
// the _tsq will be freed outside of this parser
Parser::~Parser() {}

// Backup transfer from _store to _tsq
void Parser::backup() {
	void *ptr = _store.top();
	_store.pop();
	_tsq->push_front(ptr);
}

void Parser::backup(size_t n) {
	for (size_t i = 0; i < n; i++)
		backup();
}

// Requirement function
Parser::TagPair Parser::require(LexTag ltag) {
	auto pr = get();
	std::cout << "REQ-GOT Tag: " << strlex[pr.tag] << std::endl;

	if (pr.tag != ltag)
		throw bad_tag(pr.tag, ltag);
	
	return pr;
}

// Try grammar function
bool Parser::try_grammar(VTags &tags, const std::vector <LexTag> &codes) {
	for (LexTag c : codes) {
		auto pr = get();
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

// Getter helper
Parser::TagPair Parser::get() {
	if (_tsq->empty())
		throw eoq();

	void *ptr = _tsq->pop();
	_store.push(ptr);
	return {ptr, get_ltag(ptr)};
}

// Parse algorithms
void Parser::alg() {
	std::string ident;
	Args args;
	
	// TODO: need a get function that checks for empty-ness
	TagPair pr = require(IDENTIFIER);
	ident = Identifier::cast(pr.data);

	require(LPAREN);

	// Loop through all available parameters
	do {
		pr = get();
		if (pr.tag == IDENTIFIER) {
			args.push_back(Identifier::cast(pr.data));

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

// Parse statement
void Parser::statement() {
	VTags vt;

	if (try_grammar(vt, {IDENTIFIER, ASSIGN_EQ})) {
		std::cout << "\tlooking for an expression now..." << std::endl;
		std::cout << "\tident was " << Identifier::cast(vt[0].data) << std::endl;
	}
}

// Final run function
void Parser::run() {
	while (!_tsq->empty()) {
		auto pr = get();

		std::cout << "Tag = " << strlex[pr.tag] << std::endl;

		if (pr.tag == ALGORITHM) {
			std::cout << "ALGORITHM!!" << std::endl;
			alg();
		} /* else if (pr.tag == IDENTIFIER) {
			cout << "STATEMENT!!" << endl;
			// statement
			backup();
			statement();
		} */
	}
	
	std::cout << "Done." << std::endl;
}

}