#ifndef VM_H_
#define VM_H_

// C/C++ headers
#include <vector>
#include <cstdlib>

// Engine headers
#include <token.hpp>

namespace zhetapi {

/* The max number is occupied by summation and integration, which are both not
 * impleneted yet.
 */
#define MAX_OPERANDS    4

enum class OpCodes : uint8_t {
        op_set, // [addr] op1
        op_jmp  // [addr]
};

struct instruction {
        uint8_t size;
        void *  data;
};

struct vm {
        // Program counter
        size_t                          pc;

        std::vector <Token *>           ram;
        std::vector <instruction>       iram;

        // 0th bit is CMP
        uint8_t                         flags;

        /* Constant registers:
         *
         * Maximum of 4 corresponds to the maximum number of operands for any
         * operation. There should not be any need for more.
         */
        OpZ     Zreg[MAX_OPERANDS];
        OpQ     Qreg[MAX_OPERANDS];
        OpR     Rreg[MAX_OPERANDS];
};

}

#endif
