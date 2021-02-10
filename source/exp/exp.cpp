#include <image.hpp>
#include <std/filters.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	image::Image img = image::load_png("zhetapi-logo.png");

	// cout << "img = " << img << endl;

	img.show();

	image::Image crop = img.crop({99, 99}, {299, 299});

	// cout << "crop = " << crop << endl;

	crop.show();

	image::Convolution <double> conv({
		{0, 1, 0},
		{0, 0, 1},
		{1, 0, 0}
	});
}
