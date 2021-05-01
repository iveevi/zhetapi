#ifndef FEEDER_H_
#define FEEDER_H_

// C/C++ headers
#include <fstream>
#include <mutex>
#include <string>

namespace zhetapi {

// Smart source handler
struct Source {
	std::string *	src	= nullptr;
	std::ifstream *	file	= nullptr;
	size_t *	count	= nullptr;
	std::mutex *	lock	= nullptr;

        Source();
	Source(const Source &);
	
        Source(const std::string &);

	~Source();
};

class Feeder {
	Source	_source;
	size_t	_index = 0;

	Feeder(const Source &, size_t);
public:
	Feeder(const std::string &);

	Feeder split_at(size_t i) const;
};

}

#endif
