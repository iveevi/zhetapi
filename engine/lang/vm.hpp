#ifndef VM_H_
#define VM_H_

// C/C++ headers
#include <vector>
#include <cstdlib>
#include <exception>

// Engine headers
#include <token.hpp>
#include <operand.hpp>

#include <core/types.hpp>

namespace zhetapi {

/* The max number is occupied by summation and integration, which are both not
 * impleneted yet.
 */
#define MAX_REG_OPS    4

// Actually useless
enum OpCodes : uint8_t {
        op_set, // [addr] op1
        op_jmp  // [addr]
};

struct instruction {
	// Size only for writing
        uint8_t		size;
        uint8_t *	data;
};

// Registers for temporaries
template <class T>
struct tregister {
	Operand <T> regs[MAX_REG_OPS];

	// Add some member functions
};

uint32_t overload_hash(Token **);
bool overload_match(uint32_t, uint32_t);

struct vm {
        // Program counter
        size_t                          pc	= 0;

	// RAM is automatically stretched
        std::vector <Token *>           ram;
        std::vector <instruction>       iram;

        // 0th bit is CMP
        uint8_t                         flags;

        /* Constant registers:
         *
         * Maximum of 4 corresponds to the maximum number of operands for any
         * operation. There should not be any need for more.
         */
        tregister <Z>			z_reg;
        tregister <Q>			q_reg;
        tregister <R>			r_reg;

	void execute(const instruction &);

	// Exceptions
	class null_token : public std::runtime_error {
	public:
		null_token() : std::runtime_error("Read null token...") {}
	};
};

// Address token
struct vm_addr : public Token {
	vm_addr(size_t);
	
	size_t	index;

	virtual Token *copy() const;
	virtual bool operator==(Token *) const;
};

set_zhp_id(vm_addr, 24);

}

#endif
