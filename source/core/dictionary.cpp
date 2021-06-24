#include "../../engine/core/dictionary.hpp"

namespace zhetapi {

// Method table
MethodTable Dictionary::mtable;

// Dictionary iterator
DictionaryIterator::DictionaryIterator(THTable::iterator itr) : _itr(itr) {}

Token *DictionaryIterator::value() const
{
	return _itr->second;
}

void DictionaryIterator::assign(Token *tptr)
{
	_itr->second = tptr;
}

Token *DictionaryIterator::copy() const
{
	return new DictionaryIterator(_itr);
}

// Dictionary
Dictionary::Dictionary() : Token(&Dictionary::mtable) {}

Dictionary::Dictionary(const THTable &thtable) : _table(thtable) {}

Dictionary::Dictionary(const TTargs &ttargs)
		: Dictionary()
{
	for (const auto &pair : ttargs)
		_table[pair.first] = pair.second;
}

bool Dictionary::present(Token *tptr) const
{
	return (_table.find(tptr) != _table.end());
}

Iterator *Dictionary::begin()
{
	return (_table.size()) ? (new DictionaryIterator(_table.begin()))
		: nullptr;
}

Iterator *Dictionary::next(Iterator *itr)
{
	DictionaryIterator *ditr = dynamic_cast <DictionaryIterator *> (itr);

	if (!itr)
		return nullptr;
	
	THTable::iterator next = std::next(ditr->_itr);
	if (next != _table.end())
		return new DictionaryIterator(next);
	
	return nullptr;
}

Token *Dictionary::index(Token *tptr, bool mref)
{
	if (mref) {
		// Fill with a null spot, then return the iterator
		_table[tptr] = nullptr;

		return new DictionaryIterator(_table.find(tptr));
	}
	
	if (!present(tptr))
		return nullptr;

	return _table[tptr];
}

uint8_t Dictionary::id() const
{
	return get_zhp_id(Dictionary);
}

Token *Dictionary::copy() const
{
	return new Dictionary(_table);
}

Token::type Dictionary::caller() const
{
	return token_dictionary;
}

std::string Dictionary::dbg_str() const
{
	std::string out = "{";

	for (auto pair = _table.begin(); pair != _table.end(); pair++) {
		out += "{" + pair->first->dbg_str() + ", "
			+ pair->second->dbg_str() + "}";

		if (std::next(pair) != _table.end())
			out += ", ";
	}

	return out + "}";
}

bool Dictionary::operator==(Token *tptr) const
{
	Dictionary *dict = dynamic_cast <Dictionary *> (tptr);

	if (!dict || dict->_table.size() != _table.size())
		return false;
	
	// TODO: maybe needs a fix?
	return (dict->_table == _table);
}

}
