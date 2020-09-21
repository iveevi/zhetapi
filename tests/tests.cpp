// C/C++ headers
#include <iostream>

#include <dlfcn.h>

using namespace std;

const char file[] = "tests/fctn.so";

int main()
{
	cout << 24 << endl;

	system("g++ tests/fctn.cpp -o tests/fctn.so -shared -fPIC");

	void *handle = dlopen(file, RTLD_NOW);

	typedef double (*fctn)(double);

	fctn f = (fctn) dlsym(handle, "F");

	void *ptr = dlsym(handle, "F");

	cout << "ptr: " << ptr << endl;

	cout << "f: " << f << endl;

	const char *dlsym_error = dlerror();
	if (dlsym_error) {
		cerr << "Cannot load symbol 'F': " << dlsym_error << '\n';
		
		dlclose(handle);
		
		return 1;
	}

	cout << (*f)(24) << endl;

	dlclose(handle);
}
