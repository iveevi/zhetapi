#include <iostream>
#include <iomanip>
#include <random>

#include <dnn.hpp>
#include <image.hpp>

#include <std/interval.hpp>

#include <std/activations.hpp>
#include <std/erfs.hpp>
#include <std/optimizers.hpp>

using namespace std;
using namespace zhetapi;

using namespace zhetapi::ml;
using namespace zhetapi::utility;

// GOAL: Return 1, based on actions 1 and 0
int main()
{
	srand(clock());

	Interval <> i = 1_I;

	DNN <> model(1, {
		Layer <> (10, new Linear <double> ()),
		Layer <> (500, new ReLU <double> ()),
		Layer <> (1000, new Linear <double> ()),
		Layer <> (250000, new Sigmoid <double> ())
	});

	double in = i.uniform();

	Vector <double> noise = model({i.uniform()});

	image::Image img(255.0 * noise, 500, 500);

	image::save_png(img, "noise.png");
}
