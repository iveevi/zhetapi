#ifndef OPERAND_H
#define OPERAND_H

#include <iostream>

namespace operands {
    typedef int def_t;

    template <typename data_t>
    class operand {
        data_t val;
    public:
        operand();
        operand(data_t);

        void set(data_t);
        data_t get() const;

        template <typename type>
        friend std::ostream &operator<< (std::ostream &os, const operand <data_t> &);

        template <typename type>
        friend std::istream &operator>> (std::istream &is, operand <data_t> &);
    };

    typedef operand <def_t> num_t;
}

#include "operand.cpp"

#endif