#ifndef NODE_REFERENCE_H_
#define NODE_REFERENCE_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <core/node.hpp>

namespace zhetapi {

	class node_reference : public Token {
	protected:
		node *		__ref;
		::std::string	__symbol;
		size_t		__index;
		bool		__var;
	public:
		node_reference(node *, const ::std::string &, size_t, bool = false);

		node *get();
		const node &get() const;

		size_t index() const;

		const ::std::string &symbol() const;

		bool is_variable() const;

		type caller() const override;
		Token *copy() const override;
		::std::string str() const override;

		virtual bool operator==(Token *) const override;

		static bool address;
	};

}

#endif
