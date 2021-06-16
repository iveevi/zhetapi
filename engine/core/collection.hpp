#ifndef COLLECTION_H_
#define COLLECTION_H_

// Engine headers
#include "interfaces/generator.hpp"
#include "interfaces/iterator.hpp"
#include "indexable.hpp"
#include "types.hpp"
#include "method_table.hpp"

namespace zhetapi {

/**
 * @brief Iterator for the Collection primitive type.
 */
class CollectionIterator : public Iterator {
	Targs::iterator	_itr;
public:
	CollectionIterator(Targs::iterator);

	Token *value() const override;
	void assign(Token *) override;

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
	virtual Iterator *begin() override;
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

	// Static method table
	// static MethodTable mtable;
};

}

#endif
