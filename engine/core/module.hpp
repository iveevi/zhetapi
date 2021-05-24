#ifndef MODULE_H_
#define MODULE_H_

// Engine headers
#include "../token.hpp"

namespace zhetapi {

using NamedToken = std::pair <std::string, Token *>;

class Module : public Token {
	std::string	_name;
public:
	Module(const std::string &);
	Module(const std::string &, const std::vector <NamedToken> &);

	// Methods
	void add(const NamedToken &);

	// Virtual functions
	virtual type caller() const;
	virtual uint8_t id() const;
	virtual std::string dbg_str() const;
	virtual Token *copy() const;
	virtual bool operator==(Token *) const;
};

}

#endif