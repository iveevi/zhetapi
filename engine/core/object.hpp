#ifndef STRUCT_H_
#define STRUCT_H_

// C/C++ headers
#include <cstdint>
#include <unordered_map>

// Engine headers
#include "method_table.hpp"
#include "primitive.hpp"

namespace zhetapi {

// TODO: need some global object (vector) to hold the size for each type id

// Number of special operations
#define NSPOPS 5

// Object type
struct Object {
        // ID header
        TypeId		id;

        // Data
        void *		data;
        size_t		size;

	// Fixed size table for special operations
	// as an array of function pointers
	void *		spops[NSPOPS];

	// Indexes for special operations
	enum Spidx : size_t {
		sp_functor,		// Functor call
		sp_index,		// Indexable
		sp_present,		// Set (in)
		sp_begin,		// Generator (in) begin
		sp_end			// Generator (in) end
	};

        // Methods and members
        MethodTable *	mtable;
        size_t *	memoffs;	// TODO: also need some information on the type of each member (id)
        size_t		nmems;

	// "Inheritance" checks
	inline bool is_functor() const;
	inline bool is_indexable() const;
	inline bool is_set() const;
	inline bool is_generator() const;

	// ID assignment counter (starts from the last primitive id)
	static size_t nid;

	static size_t get_nid();

	// Hash table from type name to id (for structs)
	// TODO: add an alias from symtab
	static std::unordered_map <std::string, TypeId> idtable;
};

}

#endif