namespace trees {
	// Beginning of ttwrapper class
	template <typename data_t>
	class ttwrapper {
		operand <data_t> *oper_t;
		operation <operand <data_t>> *opn_t;
	public:
		ttwrapper();
		ttwrapper(operand <data_t>);
                ttwrapper(operand <data_t> *);
		ttwrapper(operation <operand <data_t>>);
                ttwrapper(operation <operand <data_t>> *);
                ttwrapper(const ttwrapper <data_t> &);

		operand <data_t> *get_oper() const;
		operation <operand <data_t>> *get_opn() const;

		bool operator==(operand <data_t>);
		bool operator==(operation <operand <data_t>>);

                bool operator==(const ttwrapper <data_t> &);

		template <typename T>
		friend std::ostream &operator<<(std::ostream &, const ttwrapper <T> &);

		token::type t;
	};

	template <typename data_t>
	ttwrapper <data_t> ::ttwrapper()
	{
		oper_t = nullptr;
		opn_t = nullptr;
		t = token::NONE;
	}

	template <typename data_t>
	ttwrapper <data_t> ::ttwrapper(operand <data_t> oper)
	{
		oper_t = &oper;
		opn_t = nullptr;
		t = token::OPERAND;
	}

        template <typename data_t>
	ttwrapper <data_t> ::ttwrapper(operand <data_t> *oper)
	{
		oper_t = oper;
		opn_t = nullptr;
		t = token::OPERAND;
	}

	template <typename data_t>
	ttwrapper <data_t> ::ttwrapper(operation <operand <data_t>> opn)
	{
		opn_t = &opn;
		oper_t = nullptr;
		t = token::OPERATION;
	}

        template <typename data_t>
	ttwrapper <data_t> ::ttwrapper(operation <operand <data_t>> *opn)
	{
		opn_t = opn;
		oper_t = nullptr;
		t = token::OPERATION;
	}

        template <typename data_t>
        ttwrapper <data_t> ::ttwrapper(const ttwrapper <data_t> &ttw)
        {
                t = ttw.t;
                switch (t) {
                case token::OPERAND:
                        oper_t = new operand <data_t> (*ttw.oper_t);
                        opn_t = nullptr;
                        break;
                case token::OPERATION:
                        opn_t = new operation <operand <data_t>> (*ttw.opn_t);
                        oper_t = nullptr;
                        break;
                }
        }

	template <typename data_t>
	operand <data_t> *ttwrapper <data_t> ::get_oper() const
	{
		return oper_t;
	}

	template <typename data_t>
	operation <operand <data_t>> *ttwrapper <data_t> ::get_opn() const
	{
		return opn_t;
	}

	template <typename data_t>
	bool ttwrapper <data_t> ::operator==(operand <data_t> oper)
	{
		// Throw error
		if (t != token::OPERAND)
			return false;
		return *oper_t == oper;
	}

	template <typename data_t>
	bool ttwrapper <data_t> ::operator==(operation <operand <data_t>> opn)
	{
		// Throw error
		if (t != token::OPERATION)
			return false;
		return *opn_t == opn;
	}

        template <typename data_t>
	bool ttwrapper <data_t> ::operator==(const ttwrapper <data_t> &ttw)
	{
		switch (ttw.t) {
                case token::OPERAND:
                        return *this == *(ttw.oper_t);
                case token::OPERATION:
                        return *this == *(ttw.opn_t);
                }

                return false;
	}

	template <typename data_t>
	std::ostream &operator<<(std::ostream &os, const ttwrapper <data_t> &ttw)
        {
                switch (ttw.t) {
                case token::OPERAND:
                        os << *(ttw.oper_t);
                        break;
                case token::OPERATION:
                        os << *(ttw.opn_t);
                        break;
                default:
                        os << "Undefined Wrapper Object";
                        break;
                }

                return os;
        }
}