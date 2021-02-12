#include <image.hpp>
#include <std/filters.hpp>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace zhetapi;
using namespace cv;

int main()
{
	image::Image img = image::load_png("zhetapi-logo.png");

	/*
	image::Convolution <double> conv({
		{0, 0, 0},
		{0, 1, 0},
		{0, 0, 0}
	}); */
	
	image::Convolution <double> conv({
		{-1, -1, -1},
		{-1, 8, -1},
		{-1, -1, -1}
	});

	image::Image conved = conv.process(img);

	conved.show();
}
