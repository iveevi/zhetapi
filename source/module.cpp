#include "../engine/module.hpp"
#include "../engine/core/types.hpp"
#include "../engine/core/functor.hpp"

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

Token *Module::attr(const std::string &id, Engine *ctx, const Targs &args, size_t osize)
{
	// Priorotize attributes
	if (_attributes.find(id) != _attributes.end()) {
		Token *tptr = _attributes[id];

		Functor *ftr = dynamic_cast <Functor *> (tptr);
		if (ftr && osize)
			return ftr->evaluate(ctx, args);

		return tptr;
	}

	return Token::attr(id, ctx, args, osize);
}

void Module::list_attributes(std::ostream &os) const
{
	// Attributes first
	os << "Attributes:" << std::endl;
	for (const auto &a : _attributes) {
		os << "\t" << a.first << " = "
			<< a.second->dbg_str() << std::endl;
	}

	Token::list_attributes(os);
}

void Module::add(const NamedToken &nt)
{
	add(nt.first, nt.second);
}

void Module::add(const char *name, Token *tptr)
{
	add(std::string(name), tptr);
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
