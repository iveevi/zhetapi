#ifndef TOKEN_TREE_H
#define TOKEN_TREE_H

// C++ Standard Libraries
#include <vector>

// Custom Built Libraries
#include "tree.h"
#include "node.h"
#include "defaults.h"
#include "ttwrapper.h"
#include "defaults.h"
#include "debug.h"

namespace tokens {
	template <typename data_t>
	class parser;
}

namespace trees {
	// Beginning of token_tree class
	template <typename data_t>
	class token_tree {
		node <ttwrapper <data_t>> *root;
		node <ttwrapper <data_t>> *cursor;
	public:
		// Constructors
		token_tree();
		explicit token_tree(ttwrapper <data_t> *);
		explicit token_tree(const ttwrapper <data_t> &);
                explicit token_tree(std::vector <token *> &);
                explicit token_tree(std::string);

                node <ttwrapper <data_t>> *build(std::vector
                        <token *> &, int);

		// implement later

		// token_tree(const token_tree &);
		// implement later

		void reset_cursor();

		void set_cursor(ttwrapper <data_t> *);
		void set_cursor(const ttwrapper <data_t> &);

		void add_branch(ttwrapper <data_t> *);
		void add_branch(const ttwrapper <data_t> &);
                void add_branch(const token_tree <data_t> &);

		void move_left();
		void move_right();
		void move_up();
		void move_down(std::size_t);

		node <ttwrapper <data_t>> *current();

                node <ttwrapper <data_t>> *value();

		void print() const;
		void print(node <ttwrapper <data_t>> *, int, int) const;
	};

	// token_tree implementation
	template <typename data_t>
	token_tree <data_t> ::token_tree()
	{
		// std::cout << "[DEFAULT CONSTRUCTOR]" << std::endl;
		root = nullptr;
		cursor = nullptr;
	}

	template <typename data_t>
	token_tree <data_t> ::token_tree(ttwrapper <data_t> *tptr)
	{
		root = new node <ttwrapper <data_t> *>;

		root->parent = nullptr;
		root->dptr = tptr;
		root->leaves = nullptr;

		cursor = root;
	}

	template <typename data_t>
	token_tree <data_t> ::token_tree(const ttwrapper <data_t> &tok)
	{
		root = new node <ttwrapper <data_t> *>;
		
		root->parent = nullptr;
		// Make copy constructor for
		// all tokens
		root->dptr = new token(tok);
		root->leaves = nullptr;

		cursor = root;
	}

        template <typename data_t>
        token_tree <data_t> ::token_tree(std::vector <token *> &toks)
        {
		// std::cout << "[VECTOR CONSTRUCTOR]" << std::endl;
                root = build(toks, 0);
                cursor = root;
		// print();
		// std::cout << "Finished Construction" << std::endl;
        }

        template <typename data_t>
        token_tree <data_t> ::token_tree(std::string input)
        {
		// std::cout << "[STRING CONSTRUCTOR]" << std::endl;
                std::vector <token *> toks = parser <data_t>
                        ::get_tokens(input);
                *this = token_tree(toks);
		// std::cout << "---PRINTING---" << std::endl;
		// print();
		// std::cout << "Finished String Construction" << std::endl;
        }

        template <typename data_t>
        node <ttwrapper <data_t>> *token_tree <data_t> ::build(std::vector
                <token *> &toks, int level)
        {
                list <node <ttwrapper <data_t>>> *leaves;
                node <ttwrapper <data_t>> *out, *oa, *ob;
                ttwrapper <data_t> *ttptr, *tmp;
                token *tptr, *t, norm;
                std::size_t i, save;
                
                if (toks.size() == 0)
                        return nullptr;

                if (toks.size() == 1) {
                        switch(toks[0]->caller()) {
                        case token::OPERAND:
                                ttptr = new ttwrapper <data_t> ((operand <data_t> *) (toks[0]));
                                break;
                        case token::OPERATION:
                                ttptr = new ttwrapper <data_t> ((operation <operand
                                        <data_t>> *) (toks[0]));
                                break;
                        default:
                                ttptr = nullptr;
                                break;
                        }

                        out = new node <ttwrapper <data_t>> (ttptr);
                        return out;
                }

                tptr = &defaults <data_t> ::opers[defaults <data_t> ::NOPERS];

		///dp_msg("Pringint out toks vector");
		// stl_reveal(t, toks);

                save = 0;
                for (int i = 0; i < toks.size(); i++) {
                        t = toks[i];
			///dp_msg("t:");
			// std::cout << t << std::endl;
			// std::cout << t->str() << std::endl;
                        if (t->caller() == token::OPERATION &&
                                ((operation <operand <data_t>> *) (t))->get_order() <= 
                                ((operation <operand <data_t>> *) (tptr))->get_order())
                                tptr = t, save = i;
                }

		// dp_var(save);
		// dp_var(tptr->str());
		// stl_reveal(t, toks, [](token *t) {return t->str();});

		ttptr = new ttwrapper <data_t> ((operation <operand <data_t>> *) (tptr));

		// Perform seperate algorithm for functions
		if (((operation <operand <data_t>> *) (tptr))->get_order() ==
			operation <operand <data_t>> ::FUNC_LMAX) {
			// Since functions have highest priority
			// the stage at which they are the leading
			// operation (root of subtree) is when all
			// its nodes (operands) belong to it
			// therefore, we just need to transfer operands
			// to the operation in order of parsing
			
			// (until silent multiplication such as 3sin x)
			// only consider operands after the operation

			// dp_var(toks.size());

			out = new node <ttwrapper <data_t>> (ttptr);
			out->leaves = new list <node <ttwrapper <data_t>>>;

			leaves = out->leaves;
			for (int i = save + 1; i < toks.size(); i++) {
				// dp_msg("looping")
				// dp_var(toks[i]->str())

				if (leaves == nullptr)
					leaves = new list <node <ttwrapper <data_t>>>;

				tmp = new ttwrapper <data_t> ((operand <data_t> *) toks[i]);
				leaves->curr = new node <ttwrapper <data_t>> (tmp);
				leaves->next = nullptr;
				leaves = leaves->next;
			}

			// dp_msg("DONE")
		} else { // Operation is otherwise binary
			std::vector <token *> a, b;

			a = std::vector <token *> (toks.begin(), toks.begin() + save);

			// dp_msg("a:")
			// stl_reveal(tptr, a, [](token *t) {return t->str();});

			b = std::vector <token *> (toks.begin() + (save + 1), toks.end());

			// dp_msg("b:")
			// stl_reveal(tptr, b, [](token *t) {return t->str();});

			out = new node <ttwrapper <data_t>> (ttptr);
			oa = build(a, level + 1);
			ob = build(b, level + 1);

			out->leaves = new list <node <ttwrapper <data_t>>>;
			leaves = out->leaves;
			leaves->curr = oa;
			leaves->next = new list <node <ttwrapper <data_t>>>;
			leaves->next->curr = ob;
			leaves->next->next = nullptr;
		}

                return out;
        }

	template <typename data_t>
	void token_tree <data_t> ::reset_cursor()
	{
		cursor = root;
	}

	template <typename data_t>
	void token_tree <data_t> ::set_cursor(ttwrapper <data_t> *ttwptr)
	{
		node <ttwrapper <data_t>> *nd = cursor;

		if (ttwptr == nullptr) {
			//std::cout << "nulltpr passed" << std::endl;
			return;
		}

		if (nd == nullptr) {
			cursor = new node <ttwrapper <data_t>>;
			cursor->dptr = new ttwrapper <data_t> (*ttwptr);
			cursor->parent = nullptr;
			cursor->leaves = nullptr;
                        root = cursor;
		} else {
			cursor->dptr = ttwptr;
			cursor->parent = nd->parent;
			cursor->leaves = nd->leaves;
			delete nd;
		}
	}

	template <typename data_t>
	void token_tree <data_t> ::set_cursor(const ttwrapper <data_t> &ttw)
	{
		set_cursor(new ttwrapper <data_t> (ttw));
	}

	template <typename data_t>
	void token_tree <data_t> ::add_branch(ttwrapper <data_t> *tptr)
	{
		node <ttwrapper <data_t>> *new_node = new node <ttwrapper <data_t>>;
		new_node->dptr = new ttwrapper <data_t> (*tptr);
		new_node->leaves = nullptr;
		
		if (cursor == nullptr) { // Throw a null_cursor_exception later
			std::cout << "Cursor is null, reset and come back?";
			std::cout << std::endl;
			return;
		}

		list <node <ttwrapper <data_t>>> *cleaves = cursor->leaves;

		if (cleaves == nullptr) {
			cleaves = new list <node <ttwrapper <data_t>>>;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			cleaves->curr->parent = cursor;
                        cursor->leaves = cleaves;
		} else {
			while (cleaves->next != nullptr)
				cleaves = cleaves->next;
			cleaves->next = new list <node <ttwrapper <data_t>>>;
			cleaves = cleaves->next;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			new_node->parent = cursor;
		}
	}
	
	template <typename data_t>
	void token_tree <data_t> ::add_branch(const ttwrapper <data_t> &tok)
	{
		ttwrapper <data_t> *tptr = new ttwrapper <data_t> (tok);
		add_branch(tptr);
	}

        template <typename data_t>
        void token_tree <data_t> ::add_branch(const token_tree <data_t> &ttree)
        {
                
        }

	template <typename data_t>
	void token_tree <data_t> ::move_left()
	{
		int index;

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			//std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent->leaves)->get_index(cursor);
		cursor = (cursor->parent->leaves)->get(index - 1)->curr;
	}
	
	template <typename data_t>
	void token_tree <data_t> ::move_right()
	{
                //std::cout << "in move right function " << std::endl;
		int index;

		if (cursor == nullptr) {
			std::cout << "cursor is null, exiting from move right" << std::endl;
			return;
		}

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent->leaves)->get_index(cursor);
                //std::cout << "index: " << index << std::endl;
                //IC(cursor->parent->leaves->size());
		cursor = (cursor->parent->leaves)->get(index + 1)->curr;
		//std::cout << "cursor now @" << cursor << std::endl;
	}
	
	template <typename data_t>
	void token_tree <data_t> ::move_up()
	{
		if (cursor == nullptr) {
			std::cout << "cursor is null, exiting from move up" << std::endl;
			return;
		}

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "Cursor has no parent" << std::endl;
			return;
		}

		cursor = cursor->parent;
	}

	template <typename data_t>
	void token_tree <data_t> ::move_down(std::size_t i)
	{
		if (cursor == nullptr) {
			std::cout << "cursor is null, exiting" << std::endl;
			return;
		}

		if (cursor->leaves == nullptr) { // Thrown null_leaves_exception
			std::cout << "Cursor has no leaves" << std::endl;
			return;
		}

		cursor = (cursor->leaves)->get(i)->curr;
	}

	template <typename data_t>
	node <ttwrapper <data_t>> *token_tree <data_t> ::current()
	{
		return cursor;
	}

	// Non-Member helper function
	// for general scope things
	template <typename data_t>
        node <ttwrapper <data_t>> *getval(node <ttwrapper <data_t>> *nd)
        {
                std::vector <operand <data_t>> vals;
                node <ttwrapper <data_t>> *output;
                int index;

                output = new node <ttwrapper <data_t>>;
                // Return operand wrapper if operand,
                // returns operand wrapper value if
                // operation
                switch (nd->dptr->t) {
                case token::OPERAND:
                        return nd;
                case token::OPERATION:
                        // Gather all the operands
                        // or values (in general) into
                        // a vector
			// dp_var(nd->dptr->get_token()->str())
			// dp_var(nd->leaves->size())
                        for (index = 0; index < nd->leaves->size(); index++) {
                                // Should throw error if leaf isnt an
                                // operation or an operand
				
				//dp_var(nd->leaves->get(index)->curr->dptr->get_token()->str())
                                switch (nd->leaves->get(index)->curr->dptr->t) {
                                case token::OPERAND:
					// dp_msg("operand")
                                        vals.push_back(*(nd->leaves->get(index)
                                                ->curr->dptr->get_oper()));
                                        break;
                                case token::OPERATION:
					// dp_msg("operation")
                                        // Return value if leaf is an operation
                                        // is always an operand
                                        vals.push_back(*(getval(nd->leaves->
                                                get(index)->curr)->dptr->get_oper()));
                                        break;
                                }
                        }

			// dp_var(nd->dptr->get_opn()->compute(vals))

                        return new node <ttwrapper <data_t>> (ttwrapper <data_t>
                                (nd->dptr->get_opn()->compute(vals)));
                }

                return nullptr;
        }

        template <typename data_t>
        node <ttwrapper <data_t>> *token_tree <data_t> ::value()
        {
                // Return operand wrapper if operand,
                // returns operand wrapper value if
                // operation
                return getval(root);
        }

	template <typename data_t>
	void token_tree <data_t> ::print() const
	{
                //std::cout << std::endl << "-------------" << std::endl;
                std::cout << "PRINTING TREE" << std::endl;
		print(root, 1, 0);
                //std::cout << std::endl << "ADDRESSES" << std::endl;
		//std::cout << "root @" << root << std::endl;
		//std::cout << "cursor @" << cursor << std::endl;
                //if (cursor != nullptr)
                //        IC(cursor->dptr);
                //if (root != nullptr)
                //        IC(root->dptr);
                //std::cout << "-------------" << std::endl;
	}

	template <typename data_t>
	void token_tree <data_t> ::print(node <ttwrapper <data_t>> *nd,
			int num, int lev) const
	{
                //std::cout << "Inside the print function, num = " << num;
                //std::cout << " and lev = " << lev << std::endl;
		if (nd == nullptr) 
                	return;

		int counter = lev;
		while (counter > 0) {
			std::cout << "\t";
			counter--;
		}

		std::cout << "#" << num << " - " << *(nd->dptr) << std::endl;
		//std::cout << " @" << nd << std::endl;

		list <node <ttwrapper <data_t>>> *rleaves = nd->leaves;

		counter = 0;
		while (rleaves != nullptr) {
			print(rleaves->curr, counter + 1, lev + 1);
                        rleaves = rleaves->next;
			counter++;
		}
	}
}

#endif
