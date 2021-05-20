// Engine headers
#include <lang/feeder.hpp>

namespace zhetapi {

bool is_terminal(char c)
{
	return (c == '\0')
		|| (c == EOF);
}

Source::Source() {}

// Copy constructor: only updates count
Source::Source(const Source &other)
		: file(other.file),
		src(other.src),
		count(other.count),
		lock(other.lock)
{
	lock->lock();

	// Add another hodler
	(*count)++;

	lock->unlock();
}

// File constructor: does not read the
// entire file immediately and instead
// loads into the string progressively
Source::Source(const std::string &path)
{
	file = new std::ifstream(path);
	src = new std::string;
	count = new size_t(1);
	lock = new std::mutex;
}

// Deconstructor: delete pointers only
// if there are no other holders
Source::~Source()
{
	if (--(*count) <= 0) {
		delete file;
		delete src;
		delete count;
		delete lock;
	}
}

// Feeder functions
void Feeder::skip_line()
{
	char c;
	while ((c = feed()) != EOF) {
		if (c == '\n')
			break;
	}
}

// TODO: optimize while keeping a cyclic buffer
void Feeder::skip_until(const std::string &str)
{
	std::string tmp;
	char c;

	for (size_t i = 0; i < str.length(); i++) {
		if ((c = feed()) == EOF)
			return;

		tmp += c;
	}

	while (true) {
		if (tmp == str)
			break;

		if ((c = feed()) == EOF)
			break;

		tmp = tmp.substr(1) + c;
	}
}

std::string Feeder::extract_quote()
{
	std::string out;
	char c;

	while ((c = feed()) != EOF) {
		if (c == '\\' && peek() == '\"') {
			feed();

			out += "\\\"";
		} else {
			if (c == '\"')
				break;

			out += c;
		}
	}

	return out;
}

// String feeder
StringFeeder::StringFeeder(const std::string &str)
		: _source(str) {}

StringFeeder::StringFeeder(const std::string &str, size_t i)
		: _source(str), _index(i) {}

char StringFeeder::feed()
{
	if (_index + 1 >= _source.length())
		return EOF;

	// TODO: check newline here
	return _source[++_index];
}

char StringFeeder::peek()
{
	if (_index + 1 >= _source.length())
		return EOF;

	// TODO: check newline here (actually dont)
	return _source[_index + 1];
}

Feeder *StringFeeder::pop_at(size_t i) const
{
	return new StringFeeder(_source, i);
}

// Source constructor
SourceFeeder::SourceFeeder(const std::string &path)
		: _source(path) {}

// Source and index constructor
SourceFeeder::SourceFeeder(const Source &source, size_t i)
		: _source(source), _index(i) {}

bool SourceFeeder::read_and_store()
{
	if (_source.file->peek() == EOF)
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

	return (*_source.src)[++_index];
}

char SourceFeeder::peek()
{
	// Catch up if needed
	// TODO: check failure
	if (_source.src->length() <= _index + 1)
		read_and_store();

	return (*_source.src)[_index + 1];
}

Feeder *SourceFeeder::pop_at(size_t i) const
{
	return new SourceFeeder(_source, i);
}

}
