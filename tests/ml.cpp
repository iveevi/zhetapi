#include <activations.h>
#include <iostream>

using namespace std;
using namespace ml;

int main()
{
	ReLU <double> act;

	cout << act(10) << endl;
}