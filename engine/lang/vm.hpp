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
class tmp_reg {
	Operand <T>	reg[MAX_REG_OPS];
	size_t		index = 0;
public:
	tmp_reg() {}

	// Read, pop and cycle
	Token *rpc(uint8_t *(&data)) {
		T value;

		memcpy(&value, data, sizeof(T));
		data += sizeof(T);
		
		reg[index] = Operand <T> (value);
		Token *tptr = &(reg[index]);

		// Cycle
		index = (index + 1) % MAX_REG_OPS;

		// Pop
		return tptr;
	}

	void content() {
		std::cout << "TMP-REG for ID " << get_zhp_id(Operand <T>) << std::endl;
		
		for (size_t i = 0; i < MAX_REG_OPS; i++)
			std::cout << "\t" << reg[i].dbg_str() << std::endl;
	}
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
	tmp_reg <Z>			z_reg;

	// reg_addr

	void execute(const instruction &);

	// Exceptions
	class null_token : public std::runtime_error {
	public:
		null_token() : std::runtime_error("Read null token...") {}
	};
};

// General address token
struct gen_addr : public Token {
	gen_addr(size_t);
	
	size_t	index;

	virtual Token *copy() const;
	virtual bool operator==(Token *) const;
};

// Memory address
struct mem_addr : public gen_addr {};

// Argument address
struct arg_addr : public gen_addr {};

// Setting zhp ids
set_zhp_id(mem_addr, 24);
set_zhp_id(arg_addr, 25);

}

#endif
