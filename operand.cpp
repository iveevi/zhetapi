namespace operands {
    template <typename data_t>
    operand <data_t> ::operand ()
    {
        val = nullptr;
    }

    template <typename data_t>
    operand <data_t> ::operand(data_t val)
    {
        this->val = &val;
    }

    template <typename data_t>
    operand <data_t> ::operand(data_t *pval)
    {
        this->val = pval;
    }

    template <typename data_t>
    data_t *operand <data_t> ::get() const
    {
        return val;
    }

    template <typename data_t>
    std::size_t operand <data_t> ::size() const
    {
        return sizeof(val);
    }

    template <typename data_t>
    std::ostream &operator<< (std::ostream &os, const operand <data_t> &right)
    {
        os << *(right->val);
        return os;
    }

    template <typename data_t>
    std::istream &operator>> (std::istream &is, operand <data_t> &right)
    {
        data_t temp;
        is >> temp;
        right->val = &temp;
        return is;
    }
}