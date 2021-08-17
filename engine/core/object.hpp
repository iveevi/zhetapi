#ifndef STRUCT_H_
#define STRUCT_H_

// C/C++ headers
#include <cstdint>
#include <unordered_map>
#include <vector>

// Engine headers
#include "primitive.hpp"

namespace zhetapi {

// TODO: need some global object (vector) to hold the size for each type id

// Number of special operations
#define NSPECIAL_OPERATIONS 5

// Object type
struct Object {
	// Type aliases
	using IdVec = std::vector <size_t>;
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
	enum Spidx : size_t {
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

	void debug() const;

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

// TYPE SPECIFICATION:
//
// Characters are just integers (value corresponds to ASCII code)

// Builtin types
Object mk_str(const char *);

}

#endif
