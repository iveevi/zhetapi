#ifndef VARIALBE_HOLDER_H_
#define VARIALBE_HOLDER_H_

// Engine headers
#include <token.hpp>

namespace zhetapi {

	struct variable_holder : public Token {

		::std::string _symbol;

		variable_holder(const ::std::string & = "");

		type caller() const override;
		Token *copy() const override;
		::std::string str() const override;

		virtual bool operator==(Token *) const override;
	};

	variable_holder::variable_holder(const ::std::string &str) :
		_symbol(str) {}

	Token::type variable_holder::caller() const
	{
		return Token::vrh;
	}

	Token *variable_holder::copy() const
	{
		return new variable_holder(_symbol);
	}

	::std::string variable_holder::str() const
	{
		return "\"" + _symbol + "\"";
	}

	bool variable_holder::operator==(Token *tptr) const
	{
		variable_holder *vcl = dynamic_cast <variable_holder *> (tptr);

		if (vcl == nullptr)
			return false;

		return vcl->_symbol == _symbol;
	}

}

#endif
