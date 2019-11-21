#ifndef OPERATION_H
#define OPERATION_H

#include <vector>
#include <string>

#include "operand.h"

namespace operations {
    using namespace operands;
    
    template <class oper_t>
    class operation : public token {
    public:
        // Typedefs
        typedef oper_t (*function)(const std::vector <oper_t> &);

        // Exception Classes
        class exception {
        public:
            std::string msg;

            exception();
            exception(std::string);

            virtual void set(std::string);
        };

        class argset_exception : public exception {
        public:
            argset_exception();
            argset_exception(int, const operation <oper_t> &);
            argset_exception(std::string);
            argset_exception(std::string, int, int);

            void set(int, const operation <oper_t> &);
            void set(std::string);
            void set(std::string, int, int);
        };

        class computation_exception : public exception {};

        // Member Functions
        operation();
        operation(std::string, function, int, const std::vector <std::string> &);

        void set(std::string, function, int, const std::vector <std::string> &);

        function get() const;
        function operator~ () const;

        oper_t compute(const std::vector <oper_t> &) const noexcept(false);
        oper_t operator() (const std::vector <oper_t> &) const noexcept(false);  
    private:
        // Real Members
        function func;
        std::string name;
        std::size_t opers;
        std::vector <std::string> symbols;
    };

    typedef operation <num_t> opn_t;
}

#include "operation.cpp"
#include "operation_specs.cpp"

#endif