namespace operations {
    template <class oper_t>
    operation <oper_t> ::operation()
    {
        func = nullptr;
    }

    template <class oper_t>
    operation <oper_t> ::operation(std::string str, function nfunc, int nopers,
        const std::vector <std::string> &nsymbols)
    {
        name = str;
        func = nfunc;
        opers = nopers;
        symbols = nsymbols;
    }

    template <class oper_t>
    void operation <oper_t> ::set(std::string str, function nfunc, int nopers,
        const std::vector <std::string> &nsymbols)
    {
        name = str;
        func = nfunc;
        opers = nopers;
        symbols = nsymbols;
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
            throw argset_exception(inputs.size(), *this);
        return (*func)(inputs);
    }

    template <class oper_t>
    oper_t operation <oper_t> ::operator() (const std::vector <oper_t> &inputs) const
        noexcept(false)
    {
        if (inputs.size() != opers)
            throw argset_exception(inputs.size(), *this);
        return (*func)(inputs);
    }
    
    // Exception (Base Exception Class) Implementation
    template <class oper_t>
    operation <oper_t> ::exception::exception() : msg("") {}

    template <class oper_t>
    operation <oper_t> ::exception::exception(std::string str) : msg(str) {}

    template <class oper_t>
    void operation <oper_t> ::exception::set(std::string str) {msg = str;}

    // Argset Exception (Derived Class From Exception) Implementation
    template <class oper_t>
    operation <oper_t> ::argset_exception::argset_exception()
        : operation <oper_t> ::exception() {}

    template <class oper_t>
    operation <oper_t> ::argset_exception::argset_exception(int actual, const operation <oper_t> &obj)
    {
        using std::to_string;
        exception::msg = obj.name + ": Expected " + to_string(obj.opers);
        exception::msg += " operands, received " + to_string(actual) + " instead.";
    }

    template <class oper_t>
    operation <oper_t> ::argset_exception::argset_exception(std::string str)
        : operation <oper_t> ::exception(str) {}
    
    template <class oper_t>
    operation <oper_t> ::argset_exception::argset_exception(std::string str, int expected, int actual)
    {
        using std::to_string;
        exception::msg = str + ": Expected " + to_string(expected);
        exception::msg += " operands, received " + to_string(actual) + "instead.";
    }

    template <class oper_t>
    void operation <oper_t> ::argset_exception::set(int actual, const operation <oper_t> &obj)
    {
        using std::to_string;
        exception::msg = obj.name + ": Expected " + to_string(obj.opers);
        exception::msg += " operands, received " + to_string(actual) + " instead.";
    }

    template <class oper_t>
    void operation <oper_t> ::argset_exception::set(std::string str) {exception::msg = str;}

    template <class oper_t>
    void operation <oper_t> ::argset_exception::set(std::string str, int expected, int actual)
    {
        using std::to_string;
        exception::msg = str + ": Expected " + to_string(expected);
        exception::msg += " operands, received " + to_string(actual) + "instead.";
    }
}
