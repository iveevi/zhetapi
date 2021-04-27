#ifndef NODE_REFERENCE_H_
#define NODE_REFERENCE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <core/node.hpp>

namespace zhetapi {

class node_reference : public Token {
protected:
	node *		_ref;
	std::string	_symbol;
	size_t		_index;
	bool		_var;
public:
	node_reference(node *, const ::std::string &, size_t, bool = false);

	node *get();
	const node &get() const;

	size_t index() const;

	const std::string &symbol() const;

	bool is_variable() const;

	type caller() const override;
	Token *copy() const override;
	std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;

	static bool address;
};

}

#endif
