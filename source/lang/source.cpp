#include <lang/feeder.hpp>

namespace zhetapi {

Source::Source() {}

// Copy constructor: only updates count
Source::Source(const Source &other)
		: file(other.file),
		src(other.src),
		count(other.count),
		lock(other.lock)
{
	lock->lock();

	// Add another hodler
	(*count)++;

	lock->unlock();
}

// File constructor: does not read the
// entire file immediately and instead
// loads into the string progressively
Source::Source(const std::string &path)
{
	file = new std::ifstream(path);
	src = new std::string;
	count = new size_t(1);
	lock = new std::mutex;
}

// Deconstructor: delete pointers only
// if there are no other holders
Source::~Source()
{
	if (--(*count) <= 0) {
		delete file;
		delete src;
		delete count;
		delete lock;
	}
}

}