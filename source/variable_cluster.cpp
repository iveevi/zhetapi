#include <core/variable_cluster.hpp>

namespace zhetapi {
	
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
