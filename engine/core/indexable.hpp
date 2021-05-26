#ifndef INDEXABLE_H_
#define INDEXABLE_H_

// Engine headers
#include "../token.hpp"

namespace zhetapi {

// Indexable interface: indexing operator
class Indexable : public virtual Token {
public:
	virtual Token *index(Token *);

	// Inherited from Token
	virtual size_t id() const override;
	virtual Token *copy() const override;
	virtual type caller() const override;
	virtual std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;
}

}

#endif