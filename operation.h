#ifndef OPERATION_H
#define OPERATION_H

#include "operand.h"

namespace operations {
    using namespace operands;
    
    template <class oper_t>
    class operation {
    public:
    };

    typedef operation <num_t> opn_t;
}

#include "operation.cpp"

#endif