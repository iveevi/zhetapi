#ifndef ALGORITHM_H_
#define ALGORITHM_H_

// C++ headers
#include <string>
#include <vector>

// Engine headers
#include <core/node_manager.hpp>

namespace zhetapi {

class Engine;
class node_manager;

// Algorithm class
class algorithm : public Token {
	std::string			__ident		= "";
	std::string			__alg		= "";

	std::vector <std::string>	__args		= {};

	node_manager			__compiled	= node_manager();
	
	void generate(Engine *, std::string str, node_manager &);
public:
	algorithm();
	algorithm(const algorithm &);
	algorithm(const std::string &,
		const std::string &,
		const std::vector <std::string> &);
	algorithm(const std::string &,
		const std::string &,
		const std::vector <std::string> &,
		const node_manager &);
	
	void compile(Engine *);	
	
	Token *execute(Engine *, const std::vector <Token *> &);

	// Put somewhere else
	std::vector <std::string> split(std::string str);
	
	const std::string &symbol() const;

	// Virtual functions
	type caller() const override;
	Token *copy() const override;
	std::string str() const override;

	virtual bool operator==(Token *) const override;
};

}

#endif
