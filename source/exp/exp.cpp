#include <image.hpp>

#include <std/interval.hpp>

using namespace std;
using namespace zhetapi::image;
using namespace zhetapi::utility;

int main()
{
	Interval <> a(1, 5);
	Interval <> b(6, 7, false);

	cout << "a: " << a << endl;
	cout << "b: " << b << endl;

	cout << "a | b: " << (a | b) << endl;

	cout << "length: " << (a | b).size() << endl;

	Image img(1000, 1000, 4, 255);

	for (size_t i = 400; i < 600; i++) {
		for (size_t j = 400; j < 600; j++)
			img.set_hex({i, j}, "#915a56");
	}

	double cx = 700;
	double cy = 700;

	double sides = 1000.0;
	double radius = 100.0;
	double turn = 2 * acos(-1) / sides;

	double angle = 0;
	for (double i = 0; i < sides; i++, angle += turn) {
		double px = cx + radius * cos(angle);
		double py = cy + radius * sin(angle);

		img.set_hex({px, py}, "#915a56");
	}

        img.show();
}
