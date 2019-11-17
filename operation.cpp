namespace operations {
    template <class oper_t>
    operation <oper_t> ::operation()
    {
        func = nullptr;
    }

    template <class oper_t>
    operation <oper_t> ::operation(function func)
    {
        this->func = func;
    }

    template <class oper_t>
    void operation <oper_t> ::set(function doer)
    {
        this->func = doer;
    }

    template <class oper_t>
    typename operation <oper_t>::function operation <oper_t> ::get() const
    {
        return func;
    }

    template <class oper_t>
    oper_t operation <oper_t> ::compute(oper_t &a, oper_t &b) const
    {
        return func(a, b);
    }
}