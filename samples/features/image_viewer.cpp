#include <image.hpp>

using namespace zhetapi;
using namespace zhetapi::image;

int main()
{
	Image img = load_png("zhetapi-logo.png");

	img.show();
}
