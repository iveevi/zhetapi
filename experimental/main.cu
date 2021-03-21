#include <cuda/nvarena.cuh>

#include <iostream>

using namespace zhetapi;
using namespace std;

struct sx {
	size_t a, b, c, d;
};

int main()
{
	// 2 GB
	NVArena arena(2048);

	cout << "Initted..." << endl;

	void *p1 = arena.alloc(8);
	void *p2 = arena.alloc(8);

	cout << "p1 = " << p1 << endl;
	cout << "p2 = " << p2 << endl;

	cout << "sx-size = " << sizeof(sx) << endl;

	sx *sp1 = arena.alloc <sx> (4);
	sx *sp2 = arena.alloc <sx> (4);
	
	cout << "sp1 = " << sp1 << endl;
	cout << "sp2 = " << sp2 << endl;

	while (true) {
		// cout << "Loop..." << endl;
	}
}
