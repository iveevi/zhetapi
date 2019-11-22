#ifndef APPLICATION_H
#define APPLICATION_H

#include "operand.h"
#include "operation.h"
#include "operation_specs.h"

namespace application {
	/* Class that contains
	 * the common utilities
	 * for an expression and
	 * the application in general
	 *
	 * Supports:
	 *  - reading tokens (operands and 
	 *  operations) from strings
	 */
	template <class oper_t>
	class modules {
	public:
		const int OPERATIONS = 4;
		const operation <oper_t> op_defs[] = {
			add_op <oper_t>,
			sub_op <oper_t>,
			mult_op <oper_t>,
			div_op <oper_t>
		};

		token *next_token(std::string);
		vector <token *> read(std::string);

		std::istream &operator>> (std::ostream &, token *);
		std::istream &operator>> (std::ostream &, vector <token *> &);
	};
}

#endif
