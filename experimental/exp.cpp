#include <thread>
#include <functional>
#include <vector>

#include <vector.hpp>
#include <plot.hpp>

using namespace std;
using namespace zhetapi;

auto F1 = [](double x) {
	return sin(x) + 2 * cos(2 * x) + x/2;
};

auto F2 = [](double x) {
	return x;
};

int main()
{
	Plot plt;

	plt.show();

	plt.plot(F1);
	plt.plot(F2);

	plt.plot({5, 5});
	plt.plot({0, 0});

	int a;
	cout << "enter to stop: ";
	cin >> a;

	plt.close();
}
