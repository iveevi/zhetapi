#ifndef NODE_REFERENCE_H_
#define NODE_REFERENCE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <node.hpp>

namespace zhetapi {

	class node_reference : public token {
	protected:
		node *		__ref;
		std::string	__symbol;
		size_t		__index;
		bool		__var;
	public:
		node_reference(node *, const std::string &, size_t, bool = false);

		node *get();
		const node &get() const;

		size_t index() const;

		const std::string &symbol() const;

		bool is_variable() const;

		type caller() const override;
		token *copy() const override;
		std::string str() const override;

		virtual bool operator==(token *) const override;

		static bool address;
	};

	bool node_reference::address = true;

	node_reference::node_reference(node *ref, const std::string &str, size_t
			idx, bool var) : __ref(ref), __symbol(str),
		__index(idx), __var(var) {}

	node *node_reference::get()
	{
		return __ref;
	}

	const node &node_reference::get() const
	{
		return *__ref;
	}

	size_t node_reference::index() const
	{
		return __index;
	}

	const std::string &node_reference::symbol() const
	{
		return __symbol;
	}

	bool node_reference::is_variable() const
	{
		return __var;
	}

	token::type node_reference::caller() const
	{
		return token::ndr;
	}

	token *node_reference::copy() const
	{
		return new node_reference(__ref, __symbol, __index, __var);
	}

	std::string node_reference::str() const
	{
		std::ostringstream oss;

		if (address)
			oss << "\"" << __symbol << "\" points to " << __ref;
		else
			oss << "\"" << __symbol << "\"";

		return oss.str();
	}

	bool node_reference::operator==(token *tptr) const
	{
		node_reference *ndr = dynamic_cast <node_reference *> (tptr);

		if (ndr == nullptr)
			return false;

		return ndr->__symbol == __symbol;
	}

}

#endif
