#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <gmpxx.h>

#include "utility.h"

using namespace std;

int main()
{
	// Data Setup
	vector <pair <double, double>> D {
		{0, 0},
		{1, 3},
		{2, 4},
		{5, 5},
		{9, 14}
	};

	functor <double> model("model(a, b, x) = ax^2 + bx^(0.5)");

	vector <double> ws = utility::gradient_descent(D, {1, 1}, model, 2, 10, 500, 0.0001, 0.1, 1E-10);

	printf("\nSummary\n");
	cout << string(20, '=') << endl;
		
	for (size_t i = 0; i < ws.size(); i++)
		printf("Best ws[%lu]:\t%g\n", i, ws[i]);
}
