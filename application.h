#ifndef APPLICATION_H
#define APPLICATION_H

/* Change to a templated application */
namespace application {
    operation <num_t> operation_defs[];
    const int OPERATIONS;

    // Number of operations
    OPERATIONS = 4;

    // List of the default operations
    application::operation_defs = {
        add_op <num_t>,
        sub_op <num_t>,
        mult_op <num_t>,
        div_op <num_t>
    };

    class expression {
        
    };

    class application {
    public:        
        // Run Function For Applcation
        void run();
    };
}

#endif