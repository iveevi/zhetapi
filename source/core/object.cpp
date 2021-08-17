#include "../../engine/core/object.hpp"

// C/C++ headers
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace zhetapi {

// Static variables and methods
TypeId Object::nid = MAX_PRIMITIVE_IDS + 1;

TypeId Object::get_nid()
{
	return (nid++);
}

std::unordered_map <std::string, TypeId> Object::idtable;

// Type retrieving methods
inline bool Object::is_functor() const
{
	return (spops[sp_functor] != nullptr);
}

inline bool Object::is_indexable() const
{
	return (spops[sp_index] != nullptr);
}

inline bool Object::is_set() const
{
	return (spops[sp_present] != nullptr);
}

inline bool Object::is_generator() const
{
	return (spops[sp_begin] != nullptr)
		&& (spops[sp_end] != nullptr);
}

// Actual methods
void Object::debug() const
{
	if (dbg)
		return dbg(this);

	throw std::runtime_error("Null debugger for object");
}

// TYPE CONSTRUCTORS AND DEBUGGERS
void str_dbg(const Object *obj)
{
	std::cout << "String object (" << obj << "):"
		<< "\n\tstr = " << (const char *) obj->data
		<< "\n\tid = " << obj->id
		<< std::endl;
}

Object mk_str(const char *str)
{
	static TypeId str_id = Object::get_nid();

	return Object {
		.id = str_id,
		.data = (void *) str,
		.size = std::strlen(str),
		.spops = {
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr
		},
		.methods = {},
		.meminfo = {},
		.dbg = str_dbg
	};
}

}
