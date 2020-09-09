#ifndef NODE_REFERENCE_H_
#define NODE_REFERENCE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <node.hpp>

namespace zhetapi {

	class node_reference : public token {
		node *		__ref;
		std::string	__symbol;
	public:
		node_reference(node *, const std::string &);

		node *get();
		const node &get() const;

		const std::string &symbol() const;

		type caller() const override;
		token *copy() const override;
		std::string str() const override;

		virtual bool operator==(token *) const override;
	};

	node_reference::node_reference(node *ref, const std::string &str) :
		__ref(ref), __symbol(str) {}

	node *node_reference::get()
	{
		return __ref;
	}

	const node &node_reference::get() const
	{
		return *__ref;
	}

	const std::string &node_reference::symbol() const
	{
		return __symbol;
	}

	token::type node_reference::caller() const
	{
		return token::ndr;
	}

	token *node_reference::copy() const
	{
		return new node_reference(__ref, __symbol);
	}

	std::string node_reference::str() const
	{
		std::ostringstream oss;

		oss << "\"" << __symbol << "\" points to " << __ref;

		return oss.str();
	}

	bool node_reference::operator==(token *tptr) const
	{
		node_reference *ndr = dynamic_cast <node_reference *> (tptr);

		if (ndr == nullptr)
			return false;

		return ndr->__ref == __ref;
	}

}

#endif
