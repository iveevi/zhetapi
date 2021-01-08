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
	T	__val = T();
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
	std::string str() const override;
	Token *copy() const override;

	bool operator==(Token *) const override;
};

// Constructors
template <class T>
Operand <T> ::Operand () {}

template <class T>
Operand <T> ::Operand(const T &data) : __val(data) {}

template <class T>
Operand <T> ::Operand(const Operand &other) : __val(other.__val) {}

template <class T>
Operand <T> &Operand <T> ::operator=(const Operand &other)
{
	if (this != &other)
		__val = other.__val;

	return *this;
}

// Getters and setters
template <class T>
T &Operand <T> ::get()
{
	return __val;
}

template <class T>
const T &Operand <T> ::get() const
{
	return __val;
}

template <class T>
void Operand <T> ::set(const T &x)
{
	__val = x;
}

// Virtual overrides
template <class T>
Token::type Operand <T> ::caller() const
{
	return opd;
}

template <class T>
std::string Operand <T> ::str() const
{
	std::ostringstream oss;

	oss << __val;

	return oss.str();
}

template <class T>
Token *Operand <T> ::copy() const
{
	return new Operand(__val);
}

template <class T>
bool Operand <T> ::operator==(Token *tptr) const
{	
	Operand *opd = dynamic_cast <Operand *> (tptr);

	if (opd == nullptr)
		return false;

	return (opd->__val == __val);
}

}

#endif
