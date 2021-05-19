#ifndef SET_H_
#define SET_H_

// C/C++ headers
#include <function>
#include <string>
#include <utility>

// Engine headers
#include "../token.hpp"

namespace zhetapi {

class Set : public Token {
public:
	using Permit = std::function <bool (Token *)>;
private:
	std::string	_name	= "";
	Permit		_permit;
public:
	explicit Set(const std::string &);
	Set(const std::string &, Permit);

	virtual bool present(Token *) const;

	// Inherited from Token
	virtual size_t id() const override;
	virtual Token *copy() const override;
	virtual type caller() const override;
	virtual std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;
};

class MultiSet : public Token {
public:
	// Bool is for negation
	using OpVec = std::vector <std::pair <Set *, bool>>;

	enum class Operation {
		Union,
		Intersection
	};
private:
	OpVec		_vec;
	Operation	_op	= Union;
public:
	explicit MultiSet(const OpVec &);

	bool present(Token *) const;

	// Inherited from Token
	virtual size_t id() const override;
	virtual Token *copy() const override;
	virtual type caller() const override;
	virtual std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;
};

class Generator : public Set {
public:
	
};

class Collection : public Generator {

};

class Dictionary : public Generator {

};

class 

}

#endif
