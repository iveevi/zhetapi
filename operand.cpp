#ifndef OPERAND_CPP
#define OPERAND_CPP

namespace operands {
    template <typename data_t>
    operand <data_t> ::operand () : val(data_t()) {}

    template <typename data_t>
    operand <data_t> ::operand(data_t nval)
    {
        set(nval);
    }

    template <typename data_t>
    void operand <data_t> ::set(data_t nval)
    {
        val = nval;
    }

    template <typename data_t>
    void operand <data_t> ::operator[] (data_t nval)
    {
        val = nval;
    }
    
    template <typename data_t>
    data_t &operand <data_t> ::get()
    {
        return val;
    }

    template <typename data_t>
    const data_t &operand <data_t> ::get() const
    {
        return val;
    }

    template <typename data_t>
    data_t &operand <data_t> ::operator~ ()
    {
        return val;
    }

    template <typename data_t>
    const data_t &operand <data_t> ::operator~ () const
    {
        return val;
    }

    template <typename data_t>
    std::ostream &operator<< (std::ostream &os, const operand <data_t> &right)
    {
        os << right.get();
        return os;
    }

    template <typename data_t>
    std::istream &operator>> (std::istream &is, operand <data_t> &right)
    {
        data_t temp;
        is >> temp;
        right.set(temp);
        return is;
    }
}

#endif
