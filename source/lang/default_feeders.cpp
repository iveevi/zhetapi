#include <lang/feeder.hpp>

namespace zhetapi {

// TODO: iostream here

// String feeder
StringFeeder::StringFeeder(const std::string &str)
		: _source(str) {}

StringFeeder::StringFeeder(const std::string &str, size_t i, char end)
		: _source(str), _index(i), _end(end) {}

char StringFeeder::feed()
{
	_index++;
	if (_index >= _source.length()
		|| _source[_index] == _end)
		return EOF;

	// TODO: check newline here
	return _source[_index];
}

char StringFeeder::peek()
{
	if (_index + 1 >= _source.length()
		|| _source[_index + 1] == _end)
		return EOF;

	// TODO: check newline here (actually dont)
	return _source[_index + 1];
}

char StringFeeder::prev()
{
	if (_index == 0)
		return EOF;

	return _source[_index - 1];
}

size_t StringFeeder::tellg() const
{
	return _index;
}

void StringFeeder::set_end(char c)
{
	_end = c;
}

void StringFeeder::backup(size_t diff)
{
	_index = (diff >= _index) ? 0 : _index - diff;
}

Feeder *StringFeeder::pop_at(size_t i, char c) const
{
	return new StringFeeder(_source, i, c);
}

// Source constructor
SourceFeeder::SourceFeeder(const std::string &path)
		: _source(path) {}

// Source and index constructor
SourceFeeder::SourceFeeder(const Source &source, size_t i, char end)
		: _source(source), _index(i), _end(end) {}

bool SourceFeeder::read_and_store()
{
	if (_source.file->peek() == EOF
		|| _source.file->peek() == _end)
		return false;

	// TODO: add methods for this
	// TODO: also do not keep reading character by character
	(*_source.src) += _source.file->get();

	return true;
}

char SourceFeeder::feed()
{
	// Catch up if needed
	// TODO: check failure
	if (_source.src->length() <= _index + 1)
		read_and_store();

	char c;
	if ((c = (*_source.src)[++_index]) == _end);
		return EOF;
	
	return c;
}

char SourceFeeder::peek()
{
	// Catch up if needed
	// TODO: check failure
	if (_source.src->length() <= _index + 1)
		read_and_store();

	char c;
	if ((c = (*_source.src)[_index + 1]) == _end);
		return EOF;
	
	return c;
}

char SourceFeeder::prev()
{
	if (_index == 0)
		return EOF;

	// TODO: make nice indexing function
	return (*_source.src)[_index - 1];
}

size_t SourceFeeder::tellg() const
{
	return _index;
}

void SourceFeeder::set_end(char c)
{
	_end = c;
}

void SourceFeeder::backup(size_t diff)
{
	_index = (diff >= _index) ? 0 : _index - diff;
}

Feeder *SourceFeeder::pop_at(size_t i, char c) const
{
	return new SourceFeeder(_source, i, c);
}

}
