#include <thread>
#include <functional>
#include <vector>

#include <vector.hpp>
#include <plot.hpp>

using namespace std;
using namespace zhetapi;

auto F1 = [](double x) {
	return 100 * sin(x / 100) + 50 * cos(x / 10) + sqrt(abs(x));
};

auto F2 = [](double x) {
	return x * x;
};

int main()
{
	Plot plt;

	plt.show();

	plt.plot(F1);
	plt.plot(F2);

	plt.plot({200, 200});

	int a;
	cout << "enter to stop: ";
	cin >> a;

	plt.zoom(2);
	
	cout << "enter to stop: ";
	cin >> a;

	plt.close();
}
