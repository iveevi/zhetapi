#ifndef REGISTRATION_H_
#define REGISTRATION_H_

// C/C++ headers
#include <tuple>
#include <vector>
#include <functional>

// Engine headers
#include "functor.hpp"

// TODO: separate casting from registration
namespace zhetapi {

class Registrable : public Functor {
public:
	using Evaluator = std::function <Token *(const std::vector <Token *> &)>;
private:
	Evaluator	_ftn;

	std::string	_ident;
public:
	Registrable();
	Registrable(const Registrable &);
	Registrable(const std::string &, Evaluator);

	// TODO: get rid of this
	Token *operator()(const std::vector <Token *> &) const;

	Token *evaluate(Engine *, const std::vector <Token *> &) override;

	std::string dbg_str() const override;
	type caller() const override;
	Token *copy() const override;
	bool operator==(Token *) const override;
};

}

#endif
