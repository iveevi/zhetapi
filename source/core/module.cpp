#include <core/module.hpp>
#include <core/types.hpp>

namespace zhetapi {

Module::Module(const std::string &name)
	: _name(name) {}

Module::Module(const std::string &name, const std::vector <NamedToken> &parts)
	: _name(name)
{
	// Copy token?
	for (const NamedToken &nt : parts)
		_attributes[nt.first] = nt.second;
}

void Module::add(const NamedToken &nt)
{
	add(nt.first, nt.second);
}

void Module::add(const std::string &name, Token *tptr)
{
	// TODO: Copy token?
	_attributes[name] = tptr;
}

// Virtual functions
Token::type Module::caller() const
{
	return Token::token_module;
}

uint8_t Module::id() const
{
	return zhp_id <Module> ();
}

std::string Module::dbg_str() const
{
	return "module-\"" + _name + "\"";
}

Token *Module::copy() const
{
	Module *cpy = new Module(_name);

	// _methods contain builtin stuff
	// _attributes is specific to the module
	for (const auto &pr : _attributes)
		cpy->add(pr);
	
	return cpy;
}

bool Module::operator==(Token *tptr) const
{
	Module *mod = dynamic_cast <Module *> (tptr);

	if (mod == nullptr)
		return false;

	return mod->_name == _name;
}

}
