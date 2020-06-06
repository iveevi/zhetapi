#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <gmpxx.h>

#include "utility.h"
#include "network.h"

using namespace std;

int main()
{
	functor <double> ftr("f(x, y) = x^2 + y^2");

	cout << ftr << endl;

	functor <double> ftr_x = ftr.differentiate("x");
	functor <double> ftr_y = ftr.differentiate("y");
	
	cout << ftr_x << endl;
	cout << ftr_y << endl;

	size_t rounds = 10;

	vector <double> pars {1, 1};

	cout << endl << "Initial Guess(es):" << endl;

	for (size_t i = 0; i < ftr.ins(); i++)
		cout << ftr[i].symbol() << ": " << pars[i] << endl;

	cout << "Value: " << ftr.compute(pars) << endl;

	for (size_t i = 0; i < rounds; i++) {
		double val = ftr.compute(pars);

		if (val == 0) {
			cout << endl << "Found Root!" << endl;
			break;
		}

		double dx = ftr_x.compute(pars);
		double dy = ftr_y.compute(pars);

		if (dx == 0 || dy == 0) {
			cout << endl << "Shifting..." << endl;

			pars[0] += 10;
			pars[1] += 10;

			continue;
		}

		pars[0] -= val/dx;
		pars[1] -= val/dy;
		
		cout << endl << "Round #" << (i + 1) << endl;

		for (size_t i = 0; i < ftr.ins(); i++)
			cout << ftr[i].symbol() << ": " << pars[i] << endl;

		cout << "ftr: " << val << endl;
		cout << "ftr_x: " << dx << endl;
		cout << "ftr_y: " << dy << endl;

		cout << "Value: " << ftr.compute(pars) << endl;
	}

	cout << endl << "Result(s):" << endl;

	for (size_t i = 0; i < ftr.ins(); i++)
		cout << ftr[i].symbol() << ": " << pars[i] << endl;
	
	cout << "Value: " << ftr.compute(pars) << endl;

	element <double> elem {0, 0};

	cout << endl << "Solution:" << endl << utility::find_root(ftr, {1, 1}, 10) << endl;
}
