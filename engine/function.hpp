#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C/C++ headers
#include <ostream>
#include <cstdarg>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>

// Engine headers
#include <node_manager.hpp>

namespace zhetapi {

	/*
	 * Represents a mathematical function.
	 */
	template <class T, class U>
	class Function {
		node_manager <T, U> __manager;
	public:
		Function(const std::string &);
	};

}

#endif
