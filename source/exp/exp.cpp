// C/C++ headers
#include <ios>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>

// Engine headers
#include <image.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Tensor <unsigned char> logo = image::read_png("samples/imgs/sample_png.png");

	cout << "Post!" << endl;
	cout << "logo = " << logo << endl;
}
