#ifndef FEEDER_H_
#define FEEDER_H_

// C/C++ headers
#include <fstream>
#include <mutex>
#include <string>

// Engine headers
#include "../core/common.hpp"
#include "colors.hpp"

namespace zhetapi {

// For checking termination
bool is_terminal(char);

// Feeder abstract class
// TODO: needs a line method
class Feeder {
	// TODO: put index here

	// Source location (unnamed by default)
	std::string _location;
public:
	Feeder(const std::string & = "unnamed");

	const std::string &location() const;

	// State type
	using State = std::pair <char, int>;

	virtual size_t line() const = 0;

	// TODO: dont forget to flush with EOF
	virtual char feed() = 0;	// Reads, stores, returns and moves
	virtual char peek() = 0;	// Reads, stores, returns but does not moves
	virtual char prev() = 0;
	virtual bool done() = 0;
	
	virtual size_t tellg() const = 0;

	virtual State get_end() const = 0;
	virtual void set_end(State = {EOF, 1}) = 0;	// Reverse virtualness

	void set_end(char = EOF, int = 1);

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
	size_t		_line	= 1;

	char		_end	= EOF;

	// TODO: what is this for?
	bool		_second	= false;
	int		_count	= 1;		// For ending characters (eg. '{')
	
	StringFeeder(const std::string &, size_t, char = EOF);
public:
	StringFeeder(const std::string &, const std::string & = "unnamed");

	size_t line() const override;
	
	char feed() override;
	char peek() override;
	char prev() override;
	bool done() override;

	size_t tellg() const override;
	
	State get_end() const override;
	void set_end(State = {EOF, 1}) override;
	
	void backup(size_t) override;

	// TODO: any use?
	Feeder *pop_at(size_t i, char = EOF) const override;
};

StringFeeder file_feeder(const std::string &);

}

#endif
