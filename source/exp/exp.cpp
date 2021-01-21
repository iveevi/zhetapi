#include <image.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	image::Image img = image::load_png("zhetapi-logo.png");

	image::save_png(img, "tmp-save.png");
}