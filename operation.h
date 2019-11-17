#ifndef OPERATION_H
#define OPERATION_H

#include "operand.h"

namespace operations {
    using namespace operands;
    
    template <class oper_t>
    class operation {
    public:
        typedef oper_t (*function)(oper_t &, oper_t &);

        operation();
        operation(function);

        void set(function);
        function get() const;

        oper_t compute(oper_t &, oper_t &) const;
    private:
        function func;

    };

    typedef operation <num_t> opn_t;
}

#include "operation.cpp"
#include "operation_specs.cpp"

#endif