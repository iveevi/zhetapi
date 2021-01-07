#include <core/algorithm.hpp>

namespace zhetapi {

algorithm::algorithm(std::string ident, const std::vector <std::string> &args,
		const std;:vector <std::string> &statements) : __ident(ident),
		__args(args), __statements(statements) {}

Token::type algorithm::caller() const
{
	return Token::alg;
}

Token *algorithm::copy() const
{
	return new algorithm(__ident, __args, __statements);
}

std::string algorithm::str() const
{
	return __ident;
}

bool algorithm::operator==(Token *tptr) const
{
	algorithm *alg = dynamic_cast <algorithm *> (tptr);

	if (alg == nullptr)
		return false;

	return alg->__ident == __ident;
}

}
