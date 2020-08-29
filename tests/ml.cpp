#include <cstdlib>
#include <iostream>
#include <random>

#include <activation.hpp>

#include <std_activation_classes.hpp>

#include <matrix.hpp>
#include <vector.hpp>
#include <network.hpp>

using namespace std;
using namespace ml;

int main()
{
	srand(clock());
	

	Matrix <double> A(3, 3);

	cout << A << endl;

	A.randomize([]() {return rand()/(double) RAND_MAX;});

	cout << A << endl;

	Vector <double> x(3);

	cout << x << endl;
	
	x.randomize([]() {return 0.5 - rand()/(double) RAND_MAX;});
	
	Activation <double> *act = new Activation <double> ();

	ReLU <double> *relu = new ReLU <double> ();
	LeakyReLU <double> *lrelu = new LeakyReLU <double> (0.5);
	
	Sigmoid <double> *sig = new Sigmoid <double> ();
	ScaledSigmoid <double> *ssig = new ScaledSigmoid <double> (0.5);
	
	cout << "original vector:\t" << x << endl;
	
	cout << "default activated:\t" << x.activate(act) << endl;
	
	cout << "relu activated:\t\t" << x.activate(relu) << endl;
	cout << "relu derivative:\t" << x.activate(relu->derivative()) << endl;
	cout << "lrelu activated:\t" << x.activate(lrelu) << endl;
	cout << "lrelu derivative:\t" << x.activate(lrelu->derivative()) << endl;
	
	cout << "sigmoid activated:\t" << x.activate(sig) << endl;
	cout << "sigmoid derivative:\t" << x.activate(sig->derivative()) << endl;
	cout << "ssigmoid activated:\t" << x.activate(ssig) << endl;
	cout << "ssigmoid derivative:\t" << x.activate(ssig->derivative()) << endl;

	DeepNeuralNetwork <double> model({
		{4, new ReLU <double> ()},
		{4, new ReLU <double> ()}
	}, []() {return 0.5 - rand()/(double) RAND_MAX;});

	cout << model({1, -1, 5, -2}) << endl;

	model.randomize();

	cout << model({1, 1, 1, 1}) << endl;

	model.randomize();

	cout << model({1, 1, 1, 1}) << endl;
}
