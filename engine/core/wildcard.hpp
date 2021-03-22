#ifndef WILDCARD_H_
#define WILDCARD_H_

// Engine headers
#include <token.hpp>

#include <core/label.hpp>

namespace zhetapi {

class wildcard : public Token {
public:
	using predicate = bool (*)(lbl);
private:
	std::string	__symbol	= "";
	predicate	__pred		= nullptr;
public:
	wildcard(const std::string &, predicate);

	type caller() const override;
	Token *copy() const override;
	std::string str() const override;

	virtual bool operator==(Token *) const override;
};

}

#endif
