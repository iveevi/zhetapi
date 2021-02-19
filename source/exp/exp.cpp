#include <netnode.hpp>
#include <image.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	ml::NetNode n1;
	ml::NetNode n2;
	ml::NetNode n3;

	n1[0] << n2[0] << n3[0];

	n1.trace();

	image::Image img = image::load_png("zhetapi-logo.png");

	cout << img.size() << endl;

	Vector <double> x = img.size();

	cout << x << endl;
}
