#ifndef COLLECTION_H_
#define COLLECTION_H_

// Engine headers
#include "generator.hpp"
#include "indexable.hpp"
#include "iterator.hpp"
#include "types.hpp"

namespace zhetapi {

class CollectionIterator : public Iterator {
	size_t	_index = 0;
	Token *	_value = nullptr;
public:
	CollectionIterator(size_t, Token *);

	Token *value() const override;

	// Inherited from Token
	virtual Token *copy() const override;

	// Friend collection
	friend class Collection;
};

class Collection : public Generator, public Indexable {
	Targs	_tokens;

	Collection();
public:
	explicit Collection(const Targs &);

	// Set, Generator, Indexable
	virtual bool present(Token *) const override;
	virtual Iterator *begin() const override;
	virtual Iterator *next(Iterator *) override;
	virtual Token *index(Token *) override;

	// Inherited from Token
	virtual uint8_t id() const override;
	virtual Token *copy() const override;
	virtual type caller() const override;
	virtual std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;

	// Friend methods
	friend TOKEN_METHOD(col_at_method);
	friend TOKEN_METHOD(col_append_method);
};

}

#endif
