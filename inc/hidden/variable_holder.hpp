#ifndef VARIALBE_HOLDER_H_
#define VARIALBE_HOLDER_H_

// Engine headers
#include <token.hpp>

namespace zhetapi {

	struct variable_holder : public Token {

		std::string __symbol;

		variable_holder(const std::string & = "");

		type caller() const override;
		Token *copy() const override;
		std::string str() const override;

		virtual bool operator==(Token *) const override;
	};

	variable_holder::variable_holder(const std::string &str) :
		__symbol(str) {}

	Token::type variable_holder::caller() const
	{
		return Token::vrh;
	}

	Token *variable_holder::copy() const
	{
		return new variable_holder(__symbol);
	}

	std::string variable_holder::str() const
	{
		return "\"" + __symbol + "\"";
	}

	bool variable_holder::operator==(Token *tptr) const
	{
		variable_holder *vcl = dynamic_cast <variable_holder *> (tptr);

		if (vcl == nullptr)
			return false;

		return vcl->__symbol == __symbol;
	}

}

#endif
