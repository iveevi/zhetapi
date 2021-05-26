#ifndef COLLECTION_H_
#define COLLECTION_H_

// Engine headers
#include "generator.hpp"
#include "indexable.hpp"

namespace zhetapi {

class Collection : public Generator {
	std::vector <Token *>	_tokens;
public:
	Collection();
	explicit Collection(const std::vector <Token *> &);
};

}

#endif