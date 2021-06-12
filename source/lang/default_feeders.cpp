#include <lang/feeder.hpp>

namespace zhetapi {

// TODO: iostream here

// String feeder
StringFeeder::StringFeeder(const std::string &str)
		: _source(str) {}

StringFeeder::StringFeeder(const std::string &str, size_t i, char end)
		: _source(str), _index(i), _end(end) {}

// TODO: fix this +1 index offset
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

char StringFeeder::get_end() const
{
	return _end;
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
	return StringFeeder('\n' + str);
}

}
