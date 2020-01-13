#ifndef NODE_H
#define NODE_H

namespace trees {
	// Beginning of node class
	template <typename data_t>
	struct node {
		data_t *dptr;
		node *parent;
		list <node> *leaves;

                node();
                node(data_t);
                node(data_t *);
	};

        template <typename data_t>
        node <data_t> ::node()
        {
                dptr = nullptr;
                parent = nullptr;
                leaves = nullptr;
        }

        template <typename data_t>
        node <data_t> ::node(data_t data)
        {
                dptr = new data_t(data);
                parent = nullptr;
                leaves = nullptr;      
        }
	
	// Use this constructor to explicitly
        // use the same address as passed
        template <typename data_t>
        node <data_t> ::node (data_t *data)
        {
                dptr = data;
                parent = nullptr;
                leaves = nullptr;
        }
}

#endif