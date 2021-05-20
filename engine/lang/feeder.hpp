#ifndef FEEDER_H_
#define FEEDER_H_

// C/C++ headers
#include <fstream>
#include <mutex>
#include <string>

namespace zhetapi {

// For checking termination
bool is_terminal(char);

// Smart source handler
struct Source {
	std::string *	src	= nullptr;
	std::ifstream *	file	= nullptr;
	size_t *	count	= nullptr;
	std::mutex *	lock	= nullptr;

        Source();
	Source(const Source &);
	
	explicit Source(const std::string &);

	~Source();
};

// Feeder abstract class
// TODO: needs a line method
class Feeder {
public:
	// TODO: dont forget to flush with EOF
	virtual char feed() = 0;	// Reads, stores, returns and moves
	virtual char peek() = 0;	// Reads, stores, returns but does not moves

	// TODO: these two do not need to be pure virtual
	void skip_line();
	void skip_until(const std::string &);

	std::string extract_quote();

	virtual Feeder *pop_at(size_t i) const = 0;
};

// Feeder from string
class StringFeeder : public Feeder {
	std::string	_source;
	size_t		_index = 0;
	
	StringFeeder(const std::string &, size_t);
public:
	StringFeeder(const std::string &);
	
	char feed() override;
	char peek() override;

	Feeder *pop_at(size_t i) const override;
};

// Feeder from file
class SourceFeeder : public Feeder {
	Source	_source;
	size_t	_index = 0;

	SourceFeeder(const Source &, size_t);

	bool read_and_store();
public:
	SourceFeeder(const std::string &);

	char feed() override;
	char peek() override;

	Feeder *pop_at(size_t i) const override;
};

}

#endif
