#ifndef ISEQ_H_
#define ISEQ_H_

// Engine headers
#include "parser.hpp"
#include "../core/primoptns.hpp"

namespace zhetapi {

// Should also be embedded as an object class
class ISeq {
public:
	// Word size
	using Address = uint16_t;
private:
	std::vector <Address>	_code;
	std::vector <Variant>	_consts;
	Variant	*		_args;
	size_t			_nargs;
public:
	ISeq(std::queue <Parser::TagPair> &, const Args &);

	// Debugging functions
	void dump();
};

}

#endif
