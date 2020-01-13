#ifndef GROUP_H
#define GROUP_H

// Custom Built Libraries
#include "token.h"
#include "token_tree.h"

namespace tokens {
	/* Beginning of grouping token */
        template <class data_t>
        class group : public token {
		/* stored tree to represent
		 * the expression inside
		 * parenthesis */
		trees::token_tree <data_t> *tree;
	public:
		/* string constructor
		(inside parenthesis) */
		group(std::string);

		/* Default construtor */
		group();

		type caller() override;
        };
}

#endif