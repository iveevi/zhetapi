#ifndef OPERAND_H_
#define OPERAND_H_

// C++ headers
#include <sstream>

// Engine headers
#include <token.hpp>

#include <core/raw_types.hpp>

// Macros to taste
#define forward_ids(type)			\
	template <>				\
	size_t Operand <type> ::id() const;

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
	size_t id() const override;
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

template <class T>
size_t Operand <T> ::id() const
{
	return 0;
}

// Forward declare specializations
template <>
std::string Operand <bool> ::dbg_str() const;

// Forward declare ID specializations
forward_ids(Z);
forward_ids(Q);
forward_ids(R);

forward_ids(B);
forward_ids(S);

forward_ids(CmpZ);
forward_ids(CmpQ);
forward_ids(CmpR);

forward_ids(VecZ);
forward_ids(VecQ);
forward_ids(VecR);

forward_ids(VecCmpZ);
forward_ids(VecCmpQ);
forward_ids(VecCmpR);

forward_ids(MatZ);
forward_ids(MatQ);
forward_ids(MatR);

forward_ids(MatCmpZ);
forward_ids(MatCmpQ);
forward_ids(MatCmpR);

}

#endif
