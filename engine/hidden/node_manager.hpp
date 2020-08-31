#ifndef NODE_MANAGER_H_
#define NODE_MADAGER_H_

#include <node.hpp>
#include <barn.hpp>

template <class T, class U>
class node_manager {
	Barn <T, U>			__brn;
	node				__tree;
	std::vector <std::string>	__params;
public:
};

#endif