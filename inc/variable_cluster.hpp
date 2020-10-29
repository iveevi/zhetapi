#ifndef VARIABLE_CLUSTER_H_
#define VARIABLE_CLUSTER_H_

// Engine headers
#include <token.hpp>

namespace zhetapi {

	struct variable_cluster : public Token {

		std::string __cluster;

		variable_cluster(const std::string & = "");

		type caller() const override;
		Token *copy() const override;
		std::string str() const override;

		virtual bool operator==(Token *) const override;
	};

	variable_cluster::variable_cluster(const std::string &str) :
		__cluster(str) {}

	Token::type variable_cluster::caller() const
	{
		return Token::vcl;
	}

	Token *variable_cluster::copy() const
	{
		return new variable_cluster(__cluster);
	}

	std::string variable_cluster::str() const
	{
		return "\"" + __cluster + "\"";
	}

	bool variable_cluster::operator==(Token *tptr) const
	{
		variable_cluster *vcl = dynamic_cast <variable_cluster *> (tptr);

		if (vcl == nullptr)
			return false;

		return vcl->__cluster == __cluster;
	}

}

#endif
