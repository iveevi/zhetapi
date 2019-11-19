namespace operations {
    template <class oper_t>
    operation <oper_t> ::operation()
    {
        func = nullptr;
    }

    template <class oper_t>
    operation <oper_t> ::operation(function nfunc, int nopers)
    {
        func = nfunc;
        opers = nopers;
    }

    template <class oper_t>
    void operation <oper_t> ::set(function nfunc, int nopers)
    {
        func = nfunc;
        opers = nopers;
    }

    template <class oper_t>
    typename operation <oper_t>::function operation <oper_t> ::get() const
    {
        return func;
    }

    template <class oper_t>
    typename operation <oper_t>::function operation <oper_t> ::operator~ () const
    {
        return func;
    }

    template <class oper_t>
    oper_t operation <oper_t> ::compute(const std::vector <oper_t> &inputs) const
        noexcept(false)
    {
        if (inputs.size() != opers)
            throw argset_exception(inputs.size());
        return (*func)(inputs);
    }

    template <class oper_t>
    oper_t operation <oper_t> ::operator() (const std::vector <oper_t> &inputs) const
        noexcept(false)
    {
        if (inputs.size() != opers)
            throw argset_exception(inputs.size());
        return (*func)(inputs);
    }
    
    // Exception (Base Exception Class) Implementation
    template <class oper_t>
    typename operation <oper_t> ::exception::exception() : msg("") {}

    template <class oper_t>
    typename operation <oper_t> ::exception::exception(std::string str) : msg(str) {}

    template <class ope_t>
    typename operation <oper_t> ::exception::set(std::string str) : {msg = str;} 

    /*template <class oper_t>
    typename operation <data_t> ::argset_exception() {}

    template <class oper_t>
    typename operation <data_t> ::argset_exception(int actual)
    {
        msg = operation <data_t> ::name + ": expected ";
        msg += operation <data_t> ::opers + " operands but received ";
        msg += actual + "instead";
    }

    template <class oper_t> ::argset_exception()*/
}
