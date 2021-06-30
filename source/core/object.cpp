#include "../../engine/core/object.hpp"

namespace zhetapi {

// Static variables and methods
size_t Object::nid = MAX_PIDS + 1;

size_t Object::get_nid()
{
	return (nid++);
}

std::unordered_map <std::string, size_t> Object::idtable;

}