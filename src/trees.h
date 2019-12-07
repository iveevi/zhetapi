#ifndef TREE_H
#define TREE_H

#include <type_traits>

#include "tokens.h"

namespace trees {
	// Using declarations
	using namespace tokens;

	// Beginning of tree class
	class tree {
	public:
		enum type {TREE, TTREE, VTREE,
			FTREE, ETREE};

		virtual type caller();
	};

	tree::type tree::caller()
	{
		return TREE;
	}

	// Beginning of the helper classes
	// such as node and list

	// Beginning of list class
	template <typename data_t>
	struct list {
		data_t *curr;
		list *next;

		std::size_t get_index(data_t *);

		list *operator()(data_t *);
		list *operator[](std::size_t);
	};

	template <typename data_t>
	std::size_t list <data_t> ::get_index(data_t *nd) {
		list *cpy = this;
		int index = 0;

		while (cpy != nullptr) {
			if (*(cpy->curr->dptr) == *(nd->dptr))
				return index;
			cpy = cpy->next;
			index ++;
		}

		return - 1;
	}

	template <typename data_t>
	list <data_t> *list <data_t> ::operator()(data_t *nd) {
		list *cpy = this;

		while (cpy != nullptr) {
			if (*(cpy->curr->tptr) == *(nd->tptr))
				return cpy;
			cpy = cpy->next;
		}

		return nullptr;
	}

	template <typename data_t>
	list <data_t> *list <data_t> ::operator[](std::size_t i) {
		list *cpy = this;
		int index = i;

		while (index >= 0) {
			if (cpy == nullptr)
				return nullptr;
			cpy = cpy->next;
			index --;
		}

		return cpy;
	}

	// Beginning of node class
	template <typename data_t>
	struct node {
		data_t dptr;
		node *parent;
		list <node> *leaves;
	};

	// Beginning of token_tree class
	template <typename data_t>
	class token_tree {
	public:
		// Typedefs
		typedef operand <data_t> oper;
		typedef operation <oper> opn;

		// Constructors
		token_tree();
		token_tree(token *);
		token_tree(const token &);

		// token_tree(std::string);
		// implement later

		// token_tree(const token_tree &);
		// implement later

		void reset_cursor();

		void add_branch(token *);
		void add_branch(const token &);

		void move_left();
		void move_right();
		void move_up();
		void move_down(std::size_t);

		node <token *> *current();

		void print() const;
		void print(node <token *> *) const;
	private:
		node <token *> *root;
		node <token *> *cursor;
	};

	// token_tree implementation
	template <typename data_t>
	token_tree <data_t> ::token_tree()
	{
		root = nullptr;
		cursor = nullptr;
	}

	template <typename data_t>
	token_tree <data_t> ::token_tree(token *tptr)
	{
		root = new node <token *>;

		root->parent = nullptr;
		root->dptr = tptr;
		root->leaves = nullptr;

		cursor = root;
	}

	template <typename data_t>
	token_tree <data_t> ::token_tree(const token &tok)
	{
		root = new node <token *>;
		
		root->parent = nullptr;
		// Make sopy constructor for
		// all tokens
		root->dptr = new token(tok);
		root->leaves = nullptr;

		cursor = root;
	}

	template <typename data_t>
	void token_tree <data_t> ::reset_cursor()
	{
		cursor = root;
	}

	template <typename data_t>
	void token_tree <data_t> ::add_branch(token *tptr)
	{
		node <token *> *new_node = new node <token *>;
		new_node->dptr = new vals(tptr);
		new_node->leaves = nullptr;

		list <node <token *>> *cleaves = cursor->leaves;

		if (cursor == nullptr) { // Throw a null_cursor_exception later
			std::cout << "Cursor is null, reset and come back?";
			std::cout << std::endl;
		}

		if (cleaves == nullptr) {
			cleaves = new list <node <token *>>;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			new_node->parent = cursor;
		} else {
			while (cleaves->next != nullptr)
				cleaves = cleaves->next;
			cleaves->next = new list <node <token *>>;
			cleaves = cleaves->next;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			new_node->parent = cursor;
		}
	}
	
	template <typename data_t>
	void token_tree <data_t> ::add_branch(const token &tok)
	{
		token *tptr = new token(tok);
		add_branch(tptr);
	}

	template <typename data_t>
	void token_tree <data_t> ::move_left()
	{
		int index;

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent->leaves)->get_index(cursor);
		cursor = ((cursor->parent)[index - 1])->curr;
	}
	
	template <typename data_t>
	void token_tree <data_t> ::move_right()
	{
		int index;

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent->leaves)->get_index(cursor);
		cursor = (*(cursor->parent->leaves))[index + 1]->curr;
	}
	
	template <typename data_t>
	void token_tree <data_t> ::move_up()
	{
		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "Cursor has no parent" << std::endl;
			return;
		}

		cursor = cursor->parent;
	}

	template <typename data_t>
	void token_tree <data_t> ::move_down(std::size_t i)
	{
		if (cursor->leaves == nullptr) { // Thrown null_leaves_exception
			std::cout << "Cursor has no leaves" << std::endl;
			return;
		}

		cursor = (cursor->leaves)[i].curr;
	}

	template <typename data_t>
	node <token *> *token_tree <data_t> ::current()
	{
		return cursor;
	}

	template <typename data_t>
	void token_tree <data_t> ::print() const
	{
		print(root, 1, 0);
	}

	template <typename data_t>
	void token_tree <data_t> ::print(node <token *> *nd, int num, int lev) const
	{
		if (nd == nullptr)
			return;

		int counter = lev;
		while (counter > 0) {
			std::cout << "\t";
			counter--;
		}

		std::cout << "#" << num << *(nd->dptr) << std::endl;

		list <node <token *>> *rleaves = rcpy->leaves;

		counter = 0;
		while (rleaves != nullptr) {
			print(rleaves->curr, counter + 1, lev + 1);
			counter++;
		}
	}
}

#endif
