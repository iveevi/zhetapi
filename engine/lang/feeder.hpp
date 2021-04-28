#ifndef FEEDER_H_
#define FEEDER_H_

// C/C++ headers
#include <string>

namespace zhetapi {

struct Source {
	std::string *	src	= nullptr;
	size_t *	count	= 0;
};

class Feeder {
	Source *_source;
public:
};

}

#endif