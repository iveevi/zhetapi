#include "../../engine/lang/lexer.hpp"

// Standard headers
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace zhetapi {

Lexer::Lexer(const std::string &str) : _source(str) {}

inline char Lexer::feed()
{
	if (_index >= _source.length())
		return EOF;

	return _source[_index++];
}

inline char Lexer::peek()
{
	if (_index + 1 >= _source.length())
		return EOF;

	return _source[_index + 1];
}

inline bool Lexer::done()
{
	return (_index >= _source.length());
}

inline void Lexer::backup(size_t n)
{
	if (n > _index)
		throw std::runtime_error("Lexer: backed-up too much");

	_index -= n;
}

inline size_t Lexer::get_code(char c)
{
	// TODO: compile time maps?
	static const std::unordered_map <char, LexTag> ltags {
		{'{', LBRACE::id},
		{'}', RBRACE::id},
		{'(', LPAREN::id},
		{')', RPAREN::id},
		{-1, DONE::id}
	};

	if (ltags.find(c) != ltags.end())
		return ltags.at(c);

	// Return offset character to avoid overlap
	// TODO: fix a constant offset
	std::cout << "c = " << (int) c << std::endl;
	return (size_t) c + 128;
}

inline size_t Lexer::get_code(const std::string &str)
{
	static const std::unordered_map <std::string, LexTag> ltags {
		{"alg", ALGORITHM::id}
	};

	if (ltags.find(str) != ltags.end())
		return ltags.at(str);

	// std::cout << "str = " << str << std::endl;
	return DONE::id;
}

inline bool Lexer::good_ident(char c)
{
	// cout << "gic = " << c << endl;
	return !isspace(c) && (isdigit(c) || isalpha(c) || c == '_');
}

inline void *Lexer::check_dual(char expect, LexTag succcess, LexTag fail)
{
	// TODO: add a current() method
	if (_source[_index] == expect) {
		feed();

		return new NormalTag {succcess};
	}

	return new NormalTag {fail};
}

// Lexing helper functions
void *Lexer::read_space()
{
	// Read until non-space character
	while ((_next = feed()) != EOF) {
		if (_next == '\n') {
			_line++;
			return new NormalTag {NEWLINE::id};
		} else if (!isspace(_next)) {
			break;
		}
	}

	return nullptr;
}

void *Lexer::read_number()
{
	// TODO: deal with negatives
	if (isdigit(_next)) {
		// TODO: check for overflow, etc
		long long int ll = (_next - '0');
		long double ld = 0;

		while (isdigit((_next = feed())))
			ll = 10 * ll + (_next - '0');

		if (_next != '.') {
			backup(1);
			return (void *) (new PrimitiveTag(Primitive(ll)));
		}

		ld = ll;
		long double depth = -1;
		while (isdigit((_next = feed()))) {
			ld += ((long double) (_next - '0')) * std::pow(10, depth);
			depth--;
		}

		// Return the double
		backup(1);
		return (void *) (new PrimitiveTag(Primitive(ld)));
	}

	return nullptr;
}

inline char escape(char c)
{
	switch (c) {
	case 'n':
		return '\n';
	case 't':
		return '\t';
	case '\"':
		return '\"';
	case '\\':
		return '\\';
	default:
		break;
	}

	return 0;
}

void *Lexer::read_string()
{
	std::string out;

	char c;
	char n;
	while ((c = feed()) != EOF) {
		if (c == '\\' && (n = escape(feed())))
			out += n;
		else if (c == '\"')
			break;
		else
			out += c;
	}

	// std::cout << "String is \"" << out << "\"" << std::endl;
	// Object *optr = new Object(mk_str(out.c_str()));
	// ObjectTag *optr = new ObjectTag(mk_str(out.c_str()), STRING);
	// std::cout << "\tTO STRING: " << ObjectTag::cast(optr).to_string() << std::endl;

	return new ObjectTag(mk_str(out.c_str()), STRING::id);
}

void *Lexer::read_spec_sym()
{
	switch (_next) {
	case '\"':
		return read_string();
	case ',':
		return new NormalTag {COMMA::id};
	case '&':
		return check_dual('&', LOGIC_AND::id, BIT_AND::id);
	case '|':
		return check_dual('|', LOGIC_OR::id, BIT_OR::id);
	case '=':
		return check_dual('=', LOGIC_EQ::id, ASSIGN_EQ::id);
	case '!':
		return check_dual('=', NEQ::id, FACTORIAL::id);
	case '>':
		return check_dual('=', GTE::id, GT::id);
	case '<':
		return check_dual('=', LTE::id, LT::id);
	case '+':
		return check_dual('=', PLUS_EQ::id, PLUS::id);
	case '-':
		return check_dual('=', MINUS_EQ::id, MINUS::id);
	case '*':
		return check_dual('=', TIMES_EQ::id, TIMES::id);
	case '/':
		// TODO: gotta check for another /
		return check_dual('=', DIVIDE_EQ::id, DIVIDE::id);
	}

	return nullptr;
}

void *Lexer::read_identifier()
{
	if ((_next == '_') || isalpha(_next)) {
		std::string ident(1, _next);
		while (good_ident(_next = feed()))
			ident += _next;
		backup(1);

		// cout << "_next = \'" << _next << '\'' << endl;
		// cout << "<ident>\"" << ident << "\"</ident>\n";
		size_t pos;
		if ((pos = get_code(ident)))
			return new NormalTag {pos};

		return (void *) (new IdentifierTag(ident));
	}

	return nullptr;
}

// Main lexing routine
void *Lexer::scan()
{
	// Check for buffer completion
	if (done())
		return (void *) DONE::id;

	// Check for number before operations
	// to account for +/- at the start
	_cache = read_space();
	_cache = _cache ?: read_number();
	_cache = _cache ?: read_spec_sym();
	_cache = _cache ?: read_identifier();

	// Returning
	if (_cache)
		return _cache;

	// Return normal symbol according to character-code table
	return new NormalTag {get_code(_next)};
}

}
