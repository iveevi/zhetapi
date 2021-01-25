#include <image.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	image::Image img1 = image::load_png("zhetapi-logo.png");

	img1.show();

	image::Image img2 = image::load_png("samples/imgs/sample_png.png");

	img2.show();
}
