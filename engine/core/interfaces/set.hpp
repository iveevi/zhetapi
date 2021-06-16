#ifndef SET_H_
#define SET_H_

// C/C++ headers
#include <functional>
#include <string>
#include <utility>
#include <vector>

// Engine headers
#include "../../token.hpp"

namespace zhetapi {

// Set interface: branch-"in"
class Set : public virtual Token {
public:
	virtual bool present(Token *) const = 0;
};

// Set class permits "in"-if operations
class DisjointSet : public Set {
public:
	using Permit = std::function <bool (Token *)>;
private:
	Permit		_permit;
public:
	explicit DisjointSet(const std::string &);
	DisjointSet(const std::string &, Permit);

	// Inherited from Token
	virtual uint8_t id() const override;
	virtual Token *copy() const override;
	// virtual Token::type caller() const override;
	virtual std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;
};

class MultiSet : public Token {
public:
	// Bool is for negation
	using OpVec = std::vector <std::pair <DisjointSet *, bool>>;

	enum class Operation {
		Union,
		Intersection
	};
private:
	OpVec		_vec;
	Operation	_op	= Operation::Union;
public:
	explicit MultiSet(const OpVec &);

	bool present(Token *) const;

	// Inherited from Token
	virtual uint8_t id() const override;
	virtual Token *copy() const override;
	// virtual Token::type caller() const override;
	virtual std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;
};

}

#endif
