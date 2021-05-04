#include <lang/vm.hpp>

namespace zhetapi {

// Specific token readers
using Handler = Token *(*)(uint8_t *(&));

// Overloads
uint32_t overload_hash(Token **args)
{
	// Args should always have 4 elements
	uint32_t t1 = args[0]->id();
	uint32_t t2 = args[1]->id();
	uint32_t t3 = args[2]->id();
	uint32_t t4 = args[3]->id();

	return t1 + (t2 << 8) + (t3 << 16) + (t4 << 24);
}

// Array of token handlers
using namespace std;

// After getting the right code
Token *get_Z(uint8_t *(&data))
{
	Z value;
	memcpy(&value, data, sizeof(Z));
	data += sizeof(Z);
	return new Operand <Z> (value);
}

// General

Token *get_token(uint8_t *(&data))
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
		return (handlers[id])(data);

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
		Token *tptr = get_token(data);

		if (!tptr)
			throw null_token();
		
		cout << "tptr = " << tptr->dbg_str() << endl;

		args[i] = tptr;
	}
}

// Virtual address
vm_addr::vm_addr(size_t i) : index(i) {}

Token *vm_addr::copy() const
{
	return new vm_addr {index};
}

bool vm_addr::operator==(Token *tptr) const
{
	vm_addr *vaptr = dynamic_cast <vm_addr *> (tptr);

	return vaptr && (index == vaptr->index);
}

}