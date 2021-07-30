#include "../engine/module.hpp"
#include "../engine/engine.hpp"
#include "../engine/core/types.hpp"
#include "../engine/core/functor.hpp"

namespace zhetapi {

/**
 * @brief Name constructor.
 * 
 * @param name the name of the resulting Module object.
 */
Module::Module(const std::string &name)
		: _name(name) {}

/**
 * @brief Name and member constructor.
 * 
 * @param name the name of the resulting Module object.
 * @param parts the members to be added to the object, as a list of pairs of
 * their names (as std::strings) and values (as Tokens).
 */
Module::Module(const std::string &name, const std::vector <NamedToken> &parts)
		: _name(name)
{
	// Copy token?
	for (const NamedToken &nt : parts)
		_attributes[nt.first] = nt.second;
}

// TODO: pass the tree to attr instead of the args and osize

/**
 * @brief Overrided attribute method for Modules. Prioritizes the members of the
 * object over its own methods.
 * 
 * @param id the name of the member or method.
 * @param ctx the execution context.
 * @param args the arguments to be provided to the member/method.
 * @param osize the original size of the tree's leaves.
 */
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

/**
 * @brief Overrides the \c list_attributes method from Tokens. Prints the
 * attributes -- the members and methods -- of the object.
 */
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

/**
 * @brief Adds a member to the object.
 * 
 * @param nt a pair containing the name and value of the member.
 */
void Module::add(const NamedToken &nt)
{
	add(nt.first, nt.second);
}

/**
 * @brief Adds a member to the object.
 * 
 * @param name the name of the member.
 * @param tptr the value of the member.
 */
void Module::add(const char *name, Token *tptr)
{
	add(std::string(name), tptr);
}

/**
 * @brief Adds a member to the object.
 * 
 * @param name the name of the member.
 * @param tptr the value of the member.
 */
void Module::add(const std::string &name, Token *tptr)
{
	// TODO: Copy token?
	_attributes[name] = tptr;
}

void Module::from_add(Engine *ctx, const Args &syms)
{
	bool all = (std::find(syms.begin(), syms.end(), "*") != syms.end());

	if (all) {
		for (const auto &attr : _attributes)
			ctx->put(attr.first, attr.second);
		
		return;
	}

	// TODO: add to common
	Pardon set_syms;
	for (const std::string &sym : syms)
		set_syms.insert(set_syms.begin(), sym);

	for (const std::string &sym : set_syms) {
		bool found = false;

		for (const auto &attr : _attributes) {
			if (sym == attr.first) {
				ctx->put(attr.first, attr.second);
				found = true;

				break;
			}
		}

		if (!found) {
			throw std::runtime_error("FIXME: Library \"" + _name
				+ "\" does not contain any member \""
				+ sym + "\"");
		}
	}
}

// Virtual functions
Token::type Module::caller() const
{
	return Token::token_module;
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
