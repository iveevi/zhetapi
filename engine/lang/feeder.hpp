#ifndef FEEDER_H_
#define FEEDER_H_

// C/C++ headers
#include <fstream>
#include <mutex>
#include <string>

// Engine headers
#include "../core/common.hpp"

namespace zhetapi {

// For checking termination
bool is_terminal(char);

// Feeder abstract class
// TODO: needs a line method
class Feeder {
	// TODO: put index here
public:
	// TODO: dont forget to flush with EOF
	virtual char feed() = 0;	// Reads, stores, returns and moves
	virtual char peek() = 0;	// Reads, stores, returns but does not moves
	virtual char prev() = 0;
	
	virtual size_t tellg() const = 0;

	virtual char get_end() const = 0;
	virtual void set_end(char = EOF) = 0;

	virtual void backup(size_t) = 0;

	// TODO: these two do not need to be pure virtual
	void skip_line();
	void skip_until(const std::string &);

	std::string extract_quote();
	std::string extract_parenthesized();

	std::pair <std::string, Args>  extract_signature();

	virtual Feeder *pop_at(size_t i, char = EOF) const = 0;
};

// Feeder from string
class StringFeeder : public Feeder {
	// TODO: use a shared pointer to save resources if needed (copy
	// constructors, etc)
	std::string	_source;
	size_t		_index	= 0;
	char		_end	= EOF;
	bool		_second	= false;
	
	StringFeeder(const std::string &, size_t, char = EOF);
public:
	StringFeeder(const std::string &);
	
	char feed() override;
	char peek() override;
	char prev() override;
	
	size_t tellg() const override;
	
	char get_end() const override;
	void set_end(char = EOF) override;
	
	void backup(size_t) override;

	// TODO: any use?
	Feeder *pop_at(size_t i, char = EOF) const override;
};

StringFeeder file_feeder(const std::string &);

}

#endif
