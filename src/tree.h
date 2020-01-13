#ifndef TREE_H
#define TREE_H

namespace trees {

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
}

#endif