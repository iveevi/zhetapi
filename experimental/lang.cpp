#include "../engine/core/operation_base.hpp"
#include "../engine/core/engine_base.hpp"
#include "../engine/timer.hpp"
#include <cstdint>

using namespace std;
using namespace zhetapi;

// Primitive struct (replaces operand along with other types)
struct Primitive {
	union {
		// + ptr to struct type??
		bool b;
		long long int i;
		long double d;
	} data;
	
	uint8_t id;
};

// Overload ID
inline constexpr uint16_t ovldid(Primitive *v1, Primitive *v2)
{
	// Guaranteed that at least v1 is not null (1+ ops per operation)
	return v1->id + ((v2 ? v2->id : 0) << 8);
}

using optn = Primitive *(*)(Primitive *, Primitive *);

using ovlbase1 = std::map <uint16_t, optn>;

struct ovl {
	uint16_t ovid;
	optn main;
};

using ovlbase2 = std::vector <ovl>;

Primitive *optn_add_int_int(Primitive *v1, Primitive *v2)
{
	return new Primitive {
		{
			.i = (v1->data.i + v2->data.i)
		},
		1
	};
}

Primitive *optn_sub_int_int(Primitive *v1, Primitive *v2)
{
	return new Primitive {
		{
			.i = (v1->data.i - v2->data.i)
		},
		1
	};
}

Primitive *optn_mul_int_int(Primitive *v1, Primitive *v2)
{
	return new Primitive {
		{
			.i = (v1->data.i * v2->data.i)
		},
		1
	};
}

Primitive *optn_div_int_int(Primitive *v1, Primitive *v2)
{
	// Gotta return a double
	return new Primitive {
		{
			.i = (v1->data.i / v2->data.i)
		},
		1
	};
}

const ovlbase1 opbase1[] {
	{
		{1 + (1 << 8), &optn_add_int_int}
	},
	{
		{1 + (1 << 8), &optn_sub_int_int}
	},
	{
		{1 + (1 << 8), &optn_mul_int_int}
	},
	{
		{1 + (1 << 8), &optn_div_int_int}
	}
};

const ovlbase2 opbase2[] {
	{
		{1 + (1 << 8), &optn_add_int_int}
	},
	{
		{1 + (1 << 8), &optn_sub_int_int}
	},
	{
		{1 + (1 << 8), &optn_mul_int_int}
	},
	{
		{1 + (1 << 8), &optn_div_int_int}
	}
};

inline Primitive *do_optn2(OpCode code, Primitive *arg1, Primitive *arg2)
{
        uint16_t id = ovldid(arg1, arg2);
        
        const ovlbase1 ob = opbase1[code];
        if (ob.find(id) == ob.end())
                throw std::runtime_error("err");

        return (ob.at(id))(arg1, arg2);
}

inline Primitive *do_optn3(OpCode code, Primitive *arg1, Primitive *arg2)
{
	for (const ovl &ov : opbase2[code]) {
		if (((ov.ovid & 0x00FF) & arg1->id)
			&& (((ov.ovid & 0xFF00) >> 8) & arg2->id))
			return ov.main(arg1, arg2);
	}

	return nullptr;
}

inline Primitive *do_optn4(OpCode code, Primitive *arg1, Primitive *arg2)
{
	const ovlbase2 *ovb = &(opbase2[code]);
	for (uint8_t i = 0; i < ovb->size(); i++) {
		if ((((*ovb)[i].ovid & 0x00FF) & arg1->id)
			&& ((((*ovb)[i].ovid & 0xFF00) >> 8) & arg2->id))
			return (*ovb)[i].main(arg1, arg2);
	}

	return nullptr;
}

// Bench program
const size_t iters = 1000;

int main()
{
	// Simple addition for now
	OpZ *o1 = new OpZ(1);
	OpZ *o2 = new OpZ(1);
	
	Primitive *p1 = new Primitive {{.i = 1}, 1};
	Primitive *p2 = new Primitive {{.i = 1}, 1};

	// Setup inputs
	Targs targs = {o1, o2};

	// Run
	Timer timer;

	engine_base ebase;

	timer.start();
	for (size_t i = 0; i < iters; i++) {
		ebase.compute("+", targs);
		ebase.compute("-", targs);
		ebase.compute("*", targs);
		ebase.compute("/", targs);
	}
	timer.stop();

	cout << "Ebase time = " << timer.dt()/1000.0 << "ms." << endl;
	
	timer.start();
	for (size_t i = 0; i < iters; i++) {
		do_optn(l_add, o1, o2);
		do_optn(l_sub, o1, o2);
		do_optn(l_mul, o1, o2);
		do_optn(l_div, o1, o2);
	}
	timer.stop();

	cout << "New-id time = " << timer.dt()/1000.0 << "ms." << endl;
	
	timer.start();
	for (size_t i = 0; i < iters; i++) {
		do_optn2(l_add, p1, p2);
		do_optn2(l_sub, p1, p2);
		do_optn2(l_mul, p1, p2);
		do_optn2(l_div, p1, p2);
	}
	timer.stop();

	cout << "Primtive v1 time = " << timer.dt()/1000.0 << "ms." << endl;
	
	timer.start();
	for (size_t i = 0; i < iters; i++) {
		do_optn3(l_add, p1, p2);
		do_optn3(l_sub, p1, p2);
		do_optn3(l_mul, p1, p2);
		do_optn3(l_div, p1, p2);
	}
	timer.stop();

	cout << "Primtive v2 time = " << timer.dt()/1000.0 << "ms." << endl;
	
	timer.start();
	for (size_t i = 0; i < iters; i++) {
		do_optn4(l_add, p1, p2);
		do_optn4(l_sub, p1, p2);
		do_optn4(l_mul, p1, p2);
		do_optn4(l_div, p1, p2);
	}
	timer.stop();

	cout << "Primtive v3 time = " << timer.dt()/1000.0 << "ms." << endl;
}
