#include <cuda/nvarena.cuh>

#include <iostream>

using namespace zhetapi;
using namespace std;

int main()
{
	// 2 GB
	NVArena arena(2048);

	cout << "Initted..." << endl;
	while (true) {
		// cout << "Loop..." << endl;
	}
}
