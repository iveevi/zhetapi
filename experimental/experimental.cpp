#include "../include/image.hpp"

using namespace zhetapi;

int main()
{
	system("mkdir -p tmp/");
	image::Image img = image::load_png("zhetapi-logo.png");
	image::save_png(img, "tmp/out.png");
}
