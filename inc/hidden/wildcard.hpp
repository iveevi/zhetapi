#ifndef WILDCARD_H_
#define WILDCARD_H_

// Engine headers
#include <token.hpp>

namespace zhetapi {

	class wildcard : public token {
		node *		__ref;
		std::string	__symbol;
	public:
		wildcard();

		type caller() const override;
		token *copy() const override;
		std::string str() const override;

		virtual bool operator==(token *) const override;
	};

	wildcard::wildcard() {}

	token::type wildcard::caller() const
	{
		return token::wld;
	}

	token *wildcard::copy() const
	{
		return new wildcard();
	}

	std::string wildcard::str() const
	{
		return "wildcard";
	}

	bool wildcard::operator==(token *tptr) const
	{
		wildcard *wld = dynamic_cast <wildcard *> (tptr);

		if (wld == nullptr)
			return false;

		return true;
	}

}

#endif
