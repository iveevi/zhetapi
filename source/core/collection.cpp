#include <core/collection.hpp>

namespace zhetapi {

// Collection iterator
CollectionIterator::CollectionIterator(size_t i, Token *tptr)
		: _index(i), _value(tptr) {}

Token *CollectionIterator::value() const
{
	return _value;
}

Token *CollectionIterator::copy() const
{
	return new CollectionIterator(_index, _value->copy());
}

// Collection
Collection::Collection(const Targs &targs)
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

Iterator *Collection::begin() const
{
	if (!_tokens.size())
		return nullptr;
	
	return new CollectionIterator(0, _tokens[0]);
}

Iterator *Collection::next(Iterator *tptr)
{
	CollectionIterator *citr = dynamic_cast <CollectionIterator *> (tptr);

	if (!citr)
		return nullptr;
	
	size_t nindex = citr->_index + 1;
	if (nindex < _tokens.size())
		return new CollectionIterator(nindex, _tokens[nindex]);
	
	return nullptr;
}

Token *Collection::index(Token *tptr)
{
	OpZ *tindex = dynamic_cast <OpZ *> (tptr);

	if (!tindex)
		return nullptr;
	
	size_t index = tindex->get();
	if (index < _tokens.size())
		return _tokens[index];

	return nullptr;
}

uint8_t Collection::id() const
{
	return get_zhp_id(Collection);
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