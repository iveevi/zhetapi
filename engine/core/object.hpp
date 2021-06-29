#ifndef STRUCT_H_
#define STRUCT_H_

// C/C++ headers
#include <cstdint>

// Engine headers
#include "method_table.hpp"

namespace zhetapi {

// TODO: need some global object (vector) to hold the size for each type id

// Number of special operations
#define NSPOPS 5

// Object type
struct Object {
        // ID header
        size_t		id;

        // Data
        void *		data;
        size_t		size;

	// Fixed size table for special operations
	// as an array of function pointers
	void *		spops[NSPOS];

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

	// ID assignment counter (starts from the last primitive id)
	static size_t nid;

	// "Inheritance" checks
	inline bool is_functor() const;
	inline bool is_indexable() const;
	inline bool is_set() const;
	inline bool is_generator() const;
};

}

#endif