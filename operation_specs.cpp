#ifndef OPERATION_SPECS_CPP
#define OPERATION_SPECS_CPP

namespace operations {
    template <typename oper_t>
    operation <oper_t> add_op = operation <oper_t>
    ([](const std::vector <oper_t> &inputs) {
        return oper_t(inputs[0].get() + inputs[1].get());
    }, 2);

    template <typename oper_t>
    operation <oper_t> sub_op = operation <oper_t>
    ([](const std::vector <oper_t> &inputs) {
        return oper_t(inputs[0].get() - inputs[1].get());
    }, 2);

    template <typename oper_t>
    operation <oper_t> mult_op = operation <oper_t>
    ([](const std::vector <oper_t> &inputs) {
        return oper_t(inputs[0].get() * inputs[1].get());
    }, 2);

    template <typename oper_t>
    operation <oper_t> div_op = operation <oper_t>
    ([](const std::vector <oper_t> &inputs) {
        return oper_t(inputs[0].get() / inputs[1].get());
    }, 2);
}

#endif
