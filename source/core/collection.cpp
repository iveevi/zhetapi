#include "../../engine/core/collection.hpp"

namespace zhetapi {

// Collection class methods

// TODO: set macro to name mangle
ZHP_TOKEN_METHOD(col_at_method)
{	
	// TODO: remove assert (and use a special one that throw mistch errs)
	// use zhp cast
	assert(args.size() == 1);
	
	Collection *cptr = dynamic_cast <Collection *> (tptr);
	OpZ *index = dynamic_cast <OpZ *> (args[0]);

	return cptr->_tokens[index->get()];
}

// Uncapped append
ZHP_TOKEN_METHOD(col_append_method)
{	
	// TODO: remove assert (and use a special one that throw mistch errs)
	
	Collection *cptr = dynamic_cast <Collection *> (tptr);

	cptr->_tokens.insert(cptr->_tokens.end(),
			args.begin(),
			args.end());

	return nullptr;
}

// Registering the methods
MethodTable Collection::mtable ({
	{"at", {&col_at_method, "NA"}},
	{"append", {&col_append_method, "NA"}}
});

// Collection iterator
CollectionIterator::CollectionIterator(Targs::iterator itr)
		: _itr(itr) {}

Token *CollectionIterator::value() const
{
	return *_itr;
}

void CollectionIterator::assign(Token *tptr)
{
	*_itr = tptr;
}

Token *CollectionIterator::copy() const
{
	return new CollectionIterator(_itr);
}

// Collection
Collection::Collection() : Token(&Collection::mtable) {}

Collection::Collection(const Targs &targs)
		: Collection()
{
	_tokens.resize(targs.size());
	for (size_t i = 0; i < targs.size(); i++)
		_tokens[i] = targs[i]->copy();
}

bool Collection::present(Token *tptr) const
{
	for (size_t i = 0; i < _tokens.size(); i++) {
		if (tokcmp(tptr, _tokens[i]))
			return true;
	}
	
	return false;
}

Iterator *Collection::begin()
{
	if (!_tokens.size())
		return nullptr;
	
	return new CollectionIterator(_tokens.begin());
}

Iterator *Collection::next(Iterator *tptr)
{
	CollectionIterator *citr = dynamic_cast <CollectionIterator *> (tptr);

	if (!citr)
		return nullptr;
	
	Targs::iterator next = std::next(citr->_itr);
	if (next != _tokens.end())
		return new CollectionIterator(next);
	
	return nullptr;
}

Token *Collection::index(Token *tptr, bool mref)
{
	OpZ *tindex = dynamic_cast <OpZ *> (tptr);

	// TODO: throw here?
	if (!tindex)
		return nullptr;

	size_t index = tindex->get();
	if (mref) {
		if (index < _tokens.size())
			return new CollectionIterator(index + _tokens.begin());
	} else {
		if (index < _tokens.size())
			return _tokens[index];
	}

	// TODO: throw here?
	return nullptr;
}

Token *Collection::copy() const
{
	return new Collection(_tokens);
}

Token::type Collection::caller() const
{
	return token_collection;
}

std::string Collection::dbg_str() const
{
	std::string out = "{";

	for (size_t i = 0; i < _tokens.size(); i++) {
		out += _tokens[i]->dbg_str();

		if (i < _tokens.size() - 1)
			out += ", ";
	}

	return out + "}";
}

bool Collection::operator==(Token *tptr) const
{
	Collection *col = dynamic_cast <Collection *> (tptr);

	if (!col || col->_tokens.size() != _tokens.size())
		return false;
	
	for (size_t i = 0; i < _tokens.size(); i++) {
		if (!tokcmp(_tokens[i], col->_tokens[i]))
			return false;
	}
	
	return true;
}

}
