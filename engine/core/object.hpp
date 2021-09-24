#ifndef STRUCT_H_
#define STRUCT_H_

// C/C++ headers
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <sstream>

// Engine headers
#include "primitive.hpp"

namespace zhetapi {

// TODO: need some global object (vector) to hold the size for each type id

// Forward declarations
class ISeq;

// Number of special operations
#define NSPECIAL_OPERATIONS 6

// Object type
struct Object {
	// Type aliases
	using IdVec = std::vector <size_t>;
	using Printer = std::string (*)(const Object *);
	using Debugger = void (*)(const Object *);

        // ID header
        TypeId 		id;

        // Data
        void *		data;
        size_t		size;

	// Fixed size table for special operations
	// as an array of function pointers
	void *		spops[NSPECIAL_OPERATIONS];

	// Indexes for special operations
	// TODO: to string method
	enum Spidx : size_t {
		sp_str,			// String print (to_string)
		sp_functor,		// Functor call
		sp_index,		// Indexable
		sp_present,		// Set (in)
		sp_begin,		// Generator (in) begin
		sp_end			// Generator (in) end
	};

	// Method table object
	struct {

	} *		methods;
	IdVec 		meminfo;

	// Debugging information only
	// TODO: enclose in macro statements
	Debugger	dbg;

	inline void debug() const;

	// Printing helper function
	inline std::string to_string() const;

	// "Inheritance" checks
	inline bool is_functor() const;
	inline bool is_indexable() const;
	inline bool is_set() const;
	inline bool is_generator() const;

	// ID assignment counter (starts from the last primitive id)
	static TypeId nid;
	static TypeId get_nid();

	// Hash table from type name to id (for structs)
	// TODO: add an alias from symtab
	static std::unordered_map <std::string, TypeId> idtable;
};

// Inline functions
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

inline void Object::debug() const
{
	if (dbg)
		return dbg(this);

	throw std::runtime_error("Null debugger for object");
}

inline std::string Object::to_string() const
{
	if (spops[sp_str])
		return ((Printer) spops[sp_str])(this);

	// TODO: make a faster version of this (maybe use fmtlib)
	std::ostringstream oss;
	oss << '<' << this << ':' << id << '>';
	return oss.str();
}

// TYPE SPECIFICATION:
//
// Characters are just integers (value corresponds to ASCII code)
//
// The following are constructors for builtin types

// String type
Object mk_str(const char *);

// Collection type
Object mk_col(Object *, size_t);

// Instruction sequence type
Object mk_iseq(ISeq *);

}

#endif
