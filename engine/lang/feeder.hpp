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

	virtual Feeder *pop_at(size_t i, char = EOF) const = 0;
};

// Feeder from string
class StringFeeder : public Feeder {
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

	Feeder *pop_at(size_t i, char = EOF) const override;
};

// Feeder from file
class SourceFeeder : public Feeder {
	Source	_source;
	size_t	_index	= 0;
	char	_end	= EOF;

	SourceFeeder(const Source &, size_t, char = EOF);

	bool read_and_store();
public:
	SourceFeeder(const std::string &);

	char feed() override;
	char peek() override;
	char prev() override;

	size_t tellg() const override;

	char get_end() const override;
	void set_end(char = EOF) override;
	
	void backup(size_t) override;

	Feeder *pop_at(size_t i, char = EOF) const override;
};

}

#endif
