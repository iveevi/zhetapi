#include <core/variable_cluster.hpp>

namespace zhetapi {
	
variable_cluster::variable_cluster(const std::string &str)
		: _cluster(str) {}

Token::type variable_cluster::caller() const
{
	return Token::vcl;
}

Token *variable_cluster::copy() const
{
	return new variable_cluster(_cluster);
}

std::string variable_cluster::dbg_str() const
{
	return "vcl-\"" + _cluster + "\"";
}

bool variable_cluster::operator==(Token *tptr) const
{
	variable_cluster *vcl = dynamic_cast <variable_cluster *> (tptr);

	if (vcl == nullptr)
		return false;

	return vcl->_cluster == _cluster;
}

}
