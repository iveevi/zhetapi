#include "../../engine/core/object.hpp"

// C/C++ headers
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace zhetapi {

// Static variables and methods
TypeId Object::nid = MAX_PRIMITIVE_IDS + 1;

TypeId Object::get_nid()
{
	return (nid++);
}

std::unordered_map <std::string, TypeId> Object::idtable;

// -------------------------------
// TYPE CONSTRUCTORS AND DEBUGGERS
// -------------------------------

// String
std::string str_out(const Object *obj)
{
	return std::string((const char *) obj->data);
}

void str_dbg(const Object *obj)
{
	static size_t width = 10;

	auto field = [](const char *fname) {
		std::ostringstream oss;

		oss << "\n" << std::setw(width)
			<< std::left
			<< fname << " | ";
		return oss.str();
	};

	std::ostringstream oss;
	oss << "String object (" << obj << "):\n";
	std::string header = oss.str();
	size_t n = header.length();
	std::string line(n, '-');

	std::cout << line << '\n' << header << line
		<< field("str") << (const char *) obj->data
		<< field("id") << obj->id
		<< field("to_string") << obj->to_string()
		<< std::endl;
}

Object mk_str(const char *str)
{
	static TypeId str_id = Object::get_nid();

	// TODO: setup the methods (including constructors)
	// and idtable here exactly ONCE

	return Object {
		.id = str_id,
		.data = (void *) str,
		.size = std::strlen(str),
		.spops = {
			(void *) str_out,
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

// Collection
void col_dbg(const Object *obj)
{
	static size_t width = 10;

	auto field = [](const std::string &fname) {
		std::ostringstream oss;

		oss << "\n" << std::setw(width)
			<< std::left
			<< fname << " | ";
		return oss.str();
	};

	std::ostringstream oss;
	oss << "Collection object (" << obj << "):\n";
	std::string header = oss.str();
	size_t n = header.length();
	std::string line(n, '-');

	std::cout << line << '\n' << header << line;

	Object *arr = (Object *) obj->data;
	for (size_t i = 0; i < obj->size; i++) {
		std::cout << field("index " + std::to_string(i + 1))
			<< arr[i].to_string();
	}

	std::cout << field("id") << obj->id
		<< field("to_string") << obj->to_string()
		<< std::endl;
}

Object mk_col(Object *arr, size_t len)
{
	static TypeId col_id = Object::get_nid();

	return Object {
		.id = col_id,
		.data = (void *) arr,
		.size = len,
		.spops = {
		},
		.methods = {},
		.meminfo = {},
		.dbg = col_dbg
	};
}

}
