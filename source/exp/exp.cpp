#include <graph/matplotlibcpp.h>

using namespace std;
using namespace zhetapi;

namespace plt = matplotlibcpp;

int main()
{
	plt::plot({1,3,2,4});
	plt::show();
}
