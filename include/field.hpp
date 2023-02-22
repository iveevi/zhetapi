#pragma once

// Standard headers
#include <concepts>

namespace zhetapi {

template <class T>
class Tensor;

#ifndef __CUDACC__

// Concept for objects which behave and interact like Matrices...
template <class T, class Type>
concept concept_field = std::is_base_of_v <Tensor <T>, Type> && \
requires (Type &a, Type &b, const Type &ct, T k) {
	// Properties
	{a.length()} -> std::convertible_to <T>;

	// Operations
	{a + b} -> std::convertible_to <Type>;
	{a - b} -> std::convertible_to <Type>;
	{k * ct} -> std::convertible_to <Type>;
	{ct * k} -> std::convertible_to <Type>;
	{ct / k} -> std::convertible_to <Type>;

	// Functions
	{min(a)} -> std::convertible_to <T>;
	{max(a)} -> std::convertible_to <T>;
};

// Derivable struct to ensure that a type behaves like a Field
template <class T, class Type>
struct Field {
	Field() {
		static_assert(
			concept_field <T, Type>,
			"Type does not behave like a Field"
		);
	}
};

#else

// Derivable struct to ensure that a type behaves like a Field
template <class T, class Type>
struct Field {
	Field() {
		static_assert(
			std::is_base_of_v <Tensor <T>, Type>,
			"Type does not behave like a Field"
		);
	}
};

#endif

}
