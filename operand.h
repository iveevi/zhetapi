#ifndef OPERAND_H
#define OPERAND_H

#include <iostream>

namespace operands {
    typedef int def_t;

    template <typename data_t>
    class operand {
        data_t *val;
    public:
        operand();
        operand(data_t);
        operand(data_t *);

        data_t *get() const;
        std::size_t size() const;

        friend std::ostream &operator<< (std::ostream &os, const operand &);
        friend std::istream &operator>> (std::istream &is, operand &);
    };

    typedef operand <def_t> num_t;
}

#include "operand.cpp"

#endif