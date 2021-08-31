#include "../../engine/lang/feeder.hpp"

namespace zhetapi {

// TODO: iostream here

// String feeder
StringFeeder::StringFeeder(const std::string &str, const std::string &loc)
		: Feeder(loc), _source(str) {}

StringFeeder::StringFeeder(const std::string &str, size_t i, char end)
		: _source(str), _index(i), _end(end) {}

size_t StringFeeder::line() const
{
	return _line;
}

// TODO: fix this +1 index offset
char StringFeeder::feed()
{
	_index++;
	
	// TODO: deal with count and stuff in the base class
	// TODO: use some sort of anti system
	if (_end == '}' && _source[_index] == '{')
		_count++;

	if (_source[_index] == _end)
		_count--;
	
	// Only increment the line number on feeds
	if (_source[_index] == '\n')
		_line++;

	if (_index >= _source.length()
		|| _count <= 0)
		return EOF;

	// TODO: check newline here
	return _source[_index];
}

char StringFeeder::peek()
{
	int tcount = _count;
	if (_end == '}' && _source[_index + 1] == '{')
		tcount++;

	if (_source[_index + 1] == _end)
		tcount--;

	if (_index + 1 >= _source.length()
		|| tcount <= 0)
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

bool StringFeeder::done()
{
	return (peek() == EOF);
}

size_t StringFeeder::tellg() const
{
	return _index;
}

Feeder::State StringFeeder::get_end() const
{
	return {_end, _count};
}

void StringFeeder::set_end(State s)
{
	_end = s.first;
	_count = s.second;
}

void StringFeeder::backup(size_t diff)
{
	_index = (diff >= _index) ? 0 : _index - diff;
}

Feeder *StringFeeder::pop_at(size_t i, char c) const
{
	return new StringFeeder(_source, i, c);
}

StringFeeder file_feeder(const std::string &fpath)
{

	std::ifstream fin(fpath);

	std::string str;

	fin.seekg(0, std::ios::end);
	str.reserve(fin.tellg());
	fin.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator <char> (fin)),
			std::istreambuf_iterator <char> ());

	// TODO: fix the offset issue
	return StringFeeder('\n' + str, fpath);
}

}
