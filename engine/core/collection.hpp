#ifndef COLLECTION_H_
#define COLLECTION_H_

// C++ headers
#include <vector>

// Engine headers
#include <token.hpp>

namespace zhetapi {

class collection : public Token {
	std::vector <Token *>	__arr;
public:
	collection();
	collection(const std::vector <Token *> &);
};

#endif
