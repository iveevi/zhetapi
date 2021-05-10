#include <lang/vm.hpp>

#define TOKEN_HANDLER(name)				\
	Token *name(vm *machine, uint8_t *(&data))

namespace zhetapi {

// Specific token readers
using Handler = Token *(*)(vm *, uint8_t *(&));

// Overloads

// An overload of 0 is universal
uint32_t overload_hash(Token **args)
{
	// Args should always have 4 elements
	uint32_t t1 = args[0]->id();
	uint32_t t2 = args[1]->id();
	uint32_t t3 = args[2]->id();
	uint32_t t4 = args[3]->id();

	return t1 + (t2 << 8) + (t3 << 16) + (t4 << 24);
}

// After getting the right code
Token *get_Z(vm *machine, uint8_t *(&data))
{
	return machine->z_reg.rpc(data);
}

// Token *get_reg_addr(vm *machine, uint8_t)

// General
Token *get_token(vm *machine, uint8_t *(&data))
{
	// Lined up according to token ids
	static size_t nhandlers = 2;
	static Handler handlers[] = {
		nullptr,
		get_Z
	};

	// Get the token id
	uint8_t id = *(data++);
	if ((id > 0) && (id < nhandlers))
		return (handlers[id])(machine, data);

	// Return nullptr on failure
	return nullptr;
}

void vm::execute(const instruction &is)
{
	using namespace std;	
	// Get the data immediately
	uint8_t *data = is.data;

	uint8_t code = *(data++);
	uint8_t nops = *(data++);

	// TODO: gotta check for nullptr
	Token *args[MAX_REG_OPS];
	for (size_t i = 0; i < nops; i++) {
		Token *tptr = get_token(this, data);

		if (!tptr)
			throw null_token();
		
		cout << "tptr = " << tptr->dbg_str() << endl;

		args[i] = tptr;
	}

	if (code == op_set) {
		cout << "SET OPERATION:" << endl;
		cout << "args = {" << endl;
		for (size_t i = 0; i < nops; i++)
			cout << "\t" << args[i]->dbg_str() << endl;
		cout << "}" << endl;
	}

	z_reg.content();
}

// Virtual address
gen_addr::gen_addr(size_t i) : index(i) {}

Token *gen_addr::copy() const
{
	return new gen_addr {index};
}

bool gen_addr::operator==(Token *tptr) const
{
	gen_addr *vaptr = dynamic_cast <gen_addr *> (tptr);

	return vaptr && (index == vaptr->index);
}

}