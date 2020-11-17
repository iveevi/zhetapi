#ifndef WILDCARD_H_
#define WILDCARD_H_

// Engine headers
#include <token.hpp>

namespace zhetapi {

	class wildcard : public Token {
		node *		__ref;
		::std::string	__symbol;
	public:
		wildcard();

		type caller() const override;
		Token *copy() const override;
		::std::string str() const override;

		virtual bool operator==(Token *) const override;
	};

	wildcard::wildcard() {}

	Token::type wildcard::caller() const
	{
		return Token::wld;
	}

	Token *wildcard::copy() const
	{
		return new wildcard();
	}

	::std::string wildcard::str() const
	{
		return "wildcard";
	}

	bool wildcard::operator==(Token *tptr) const
	{
		wildcard *wld = dynamic_cast <wildcard *> (tptr);

		if (wld == nullptr)
			return false;

		return true;
	}

}

#endif
