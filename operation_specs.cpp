#ifndef OPERATION_SPECS_CPP
#define OPERATION_SPECS_CPP

namespace operations {
    template <typename oper_t>
    operation <oper_t> add = operation <oper_t>
    ([](oper_t &a, oper_t &b) {
        return oper_t(a.get() + b.get());
    });

    template <typename oper_t>
    operation <oper_t> sub = operation <oper_t>
    ([](oper_t &a, oper_t &b) {
        return oper_t(a.get() - b.get());
    });

    template <typename oper_t>
    operation <oper_t> mult = operation <oper_t>
    ([](oper_t &a, oper_t &b) {
        return oper_t(a.get() * b.get());
    });

    template <typename oper_t>
    operation <oper_t> div = operation <oper_t>
    ([](oper_t &a, oper_t &b) {
        return oper_t(a.get() / b.get());
    });
}

#endif