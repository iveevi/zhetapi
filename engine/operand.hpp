#ifndef OPERAND_H_
#define OPERAND_H_

// C++ headers
#include <sstream>

// Engine headers
#include <token.hpp>

namespace zhetapi {

// Operand class
template <class T>
class Operand : public Token {
	T	_val = T();
public:
	Operand();
	Operand(const T &);
	Operand(const Operand &);

	Operand &operator=(const Operand &);

	T &get();
	const T &get() const;

	void set(const T &);

	// Virtual functionss
	type caller() const override;
	std::string dbg_str() const override;
	Token *copy() const override;

	bool operator==(Token *) const override;
};

// Constructors
template <class T>
Operand <T> ::Operand () {}

template <class T>
Operand <T> ::Operand(const T &data) : _val(data) {}

template <class T>
Operand <T> ::Operand(const Operand &other) : _val(other._val) {}

template <class T>
Operand <T> &Operand <T> ::operator=(const Operand &other)
{
	if (this != &other)
		_val = other._val;

	return *this;
}

// Getters and setters
template <class T>
T &Operand <T> ::get()
{
	return _val;
}

template <class T>
const T &Operand <T> ::get() const
{
	return _val;
}

template <class T>
void Operand <T> ::set(const T &x)
{
	_val = x;
}

// Virtual overrides
template <class T>
Token::type Operand <T> ::caller() const
{
	return opd;
}

template <class T>
std::string Operand <T> ::dbg_str() const
{
	std::ostringstream oss;

	oss << _val;

	return oss.str();
}

template <class T>
Token *Operand <T> ::copy() const
{
	return new Operand(_val);
}

template <class T>
bool Operand <T> ::operator==(Token *tptr) const
{
	using namespace std;
	Operand *opd = dynamic_cast <Operand *> (tptr);

	if (opd == nullptr)
		return false;

	return (opd->_val == _val);
}

// Forward declare specializations
template <>
std::string Operand <bool> ::dbg_str() const;

}

#endif
