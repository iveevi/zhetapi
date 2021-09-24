#ifndef ISEQ_H_
#define ISEQ_H_

namespace zhetapi {

// Should also be embedded as an object class
struct ISeq {
	std::vector <uint8_t>	_code;
	std::vector <Variant>	_consts;
	Variant	*		_args;
	size_t			_nargs;
};

}

#endif