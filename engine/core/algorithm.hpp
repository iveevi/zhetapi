#ifndef ALGORITHM_H_
#define ALGORITHM_H_

// C++ headers
#include <string>
#include <vector>

// Engine headers
#include <core/node_manager.hpp>


namespace zhetapi {

class Barn;
class node_manager;

// Algorithm class
class algorithm : public Token {
	std::string			__ident		= "";
	std::string			__alg		= "";

	std::vector <std::string>	__args		= {};

	node_manager			__compiled;
public:
	algorithm();
	algorithm(const std::string &,
		const std::string &,
		const std::vector <std::string> &);
	algorithm(const std::string &,
		const std::string &,
		const std::vector <std::string> &,
		const node_manager &);
	
	node_manager cmped() {return __compiled;}
	
	void compile(Barn *);
	
	Token *execute(Barn *, const std::vector <Token *> &);
	Token *execute(Barn *, std::string str, node_manager &);

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
