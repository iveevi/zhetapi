#ifndef VARIABLE_REFERENCE_H_
#define VARIABLE_REFERENCE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <node_reference.hpp>

namespace zhetapi {

	class variable_reference : public node_reference {
	public:
		variable_reference(node *, const std::string &);

		type caller() const override;
		Token *copy() const override;
		
		virtual bool operator==(Token *) const override;
	};

	variable_reference::variable_reference(node *ref, const std::string
			&str) : node_reference(ref, str) {}

	Token::type variable_reference::caller() const
	{
		return Token::vbr;
	}

	Token *variable_reference::copy() const
	{
		return new variable_reference(__ref, __symbol);
	}

	bool variable_reference::operator==(Token *tptr) const
	{
		variable_reference *vbr = dynamic_cast <variable_reference *> (tptr);

		if (vbr == nullptr)
			return false;

		return vbr->__ref == __ref;
	}

}

#endif
