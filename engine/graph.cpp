#include <fstream>

#include <GL/glut.h>

#include "functor.h"

#define	PIXELS	400.0
#define RANGE	10.0

table <double> tbl {
	variable <double> {"e", exp(1)}
};

functor <double> f("f(x) = e ^ (2 * sin x)", tbl);

int main()
{
	ofstream fout("../web/templates/data");
	for (double i = -RANGE; i <= RANGE; i += RANGE/(2 * PIXELS))
		fout << i << "\t" << f(i) << endl;
}
