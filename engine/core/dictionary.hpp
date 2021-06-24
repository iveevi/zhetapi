#ifndef DICTIONARY_H_
#define DICTIONARY_H_

// C/C++ headers
#include <stdexcept>
#include <typeindex>
#include <unordered_map>

// Engine headers
#include "interfaces/generator.hpp"
#include "interfaces/indexable.hpp"
#include "interfaces/iterator.hpp"
#include "types.hpp"
#include "method_table.hpp"

namespace zhetapi {

// Dictionary iterator
class DictionaryIterator : public Iterator {
	THTable::iterator	_itr;
public:
	DictionaryIterator(THTable::iterator);

	Token *value() const override;
	void assign(Token *) override;

	// Inherited from Token
	virtual Token *copy() const override;

	// Friend collection
	friend class Dictionary;
};

// Dictionary class
class Dictionary : public Generator, public Indexable {
	THTable _table;

	Dictionary();
public:
	explicit Dictionary(const THTable &);
	explicit Dictionary(const TTargs &);

	// Set, Generator, Indexable
	virtual bool present(Token *) const override;
	virtual Iterator *begin() override;
	virtual Iterator *next(Iterator *) override;
	virtual Token *index(Token *, bool) override;

	// Inherited from Token
	virtual uint8_t id() const override;
	virtual Token *copy() const override;
	virtual type caller() const override;
	virtual std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;

	// Static method table
	static MethodTable mtable;
};

}

#endif
