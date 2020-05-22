#include <fstream>

#include <GL/glut.h>

#include "functor.h"

table <double> tbl {
	variable <double> {"e", exp(1)},
	variable <double> {"pi", acos(0)}
};

int main()
{
	string line;

	double range;
	double pixels;

	double cx;
	double cy;

	getline(cin, line);

	cin >> range >> cx >> pixels;

	ofstream fout("../output.log");

	functor <double> f(line, tbl);

	fout << "cx: " << cx << endl;
	fout << "range: " << range << endl;

	cx *= range/pixels;

	fout << "cx [conv]: " << cx << endl;

	for (double i = -range - cx; i <= range - cx; i += range/(2 * pixels))
		cout << i << "\t" << f(i) << endl;
}
