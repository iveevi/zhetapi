#ifndef UTILITY_H_
#define UTILITY_H_

#include "element.h"

namespace utility {
	template <class T>
	std::vector <element <T>> gram_shmidt(const std::vector
			<element <T>> &span) {
		assert(span.size());

		std::vector <element <T>> basis = {span[0]};
		
		element <double> nelem;
		for (size_t i = 1; i < span.size(); i++) {
			nelem = span[i];

			for (size_t j = 0; j < i; j++) {
				nelem = nelem - (inner(span[i], basis[j])
						/ inner(basis[j], basis[j]))
						* basis[j];
			}

			basis.push_back(nelem);
		}

		return basis;
	}
	
	template <class T>
	std::vector <element <T>> gram_shmidt_normalized(const std::vector
			<element <T>> &span) {
		assert(span.size());

		std::vector <element <T>> basis = {span[0].normalize()};
	
		element <double> nelem;
		for (size_t i = 1; i < span.size(); i++) {
			nelem = span[i];

			for (size_t j = 0; j < i; j++) {
				nelem = nelem - (inner(span[i], basis[j])
						/ inner(basis[j], basis[j]))
						* basis[j];
			}

			basis.push_back(nelem.normalize());
		}

		return basis;
	}
};

#endif
