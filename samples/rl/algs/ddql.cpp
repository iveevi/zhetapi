#include "../global.hpp"

#include "rbf.hpp"

extern "C" {

// Directions
Vec dirs = acos(-1) * Vec {
	0.5, 1.0, 1.5, 2.0
};

// On-line model
ml::DNN <double> model(6, {
	ml::Layer <double> (4, new ml::ReLU <double> ()),
	ml::Layer <double> (8, new ml::Sigmoid <double> ()),
	ml::Layer <double> (4, new ml::ReLU <double> ())
});

ml::DNN <double> target;

// Replay buffer
replays prbf(1000, 500);

// Singletons
Vec Q_values;
experience e;
double discount;
size_t frames;
int i;

// Extra functions
double runit()
{
	return rand() /((double) RAND_MAX);
}

// Experiment specific functions
void init(Environment env)
{
	discount = env.gamma;
	
	// Model optimizers and loss functions
	ml::Optimizer <double> *opt = new ml::Adam <double> ();
	ml::Erf <double> *cost = new ml::MeanSquaredError <double> ();

	model.set_optimizer(opt);
	model.set_cost(cost);

	// Add a deconstruct function for freeing memory
	target = model;
}

double action(Vec S)
{
	// Get the position
	Vec P = {S[0], S[1]};

	double angle;

	e.current = S;

	// Q-value
	Q_values = model(S);

	// Get the action
	double rnd = runit();

	i = -1;
	if (rnd > 0.1) {
		i = Q_values.imax();

		e.index = i;

		angle = dirs[i];
	} else {
		angle = 2 * acos(-1) * runit();;

		i = 2 * (angle/(acos(-1)) - 0.5);

		e.index = i;
	}

	return angle;
}

void reward(double reward, Vec N, bool valid, double &error)
{
	// Update experience values
	e.reward = reward;
	e.next = N;

	// TD-error
	size_t im = model(N).imax();

	double err = fabs(reward + discount * target(N)[im] - Q_values[i]);

	e.error = err;

	// Termination?
	e.done = !valid;

	// For statistics
	error += err;
	
	// Add the experience to the buffer
	prbf.add(e);

	// Train each step (if full)
	if (prbf.full()) {
		vector <experience> batch = prbf.sample();

		DataSet <double> ins;
		DataSet <double> outs;

		for (auto e : batch) {
			ins.push_back(e.current);

			// Place this transformer into a separate function
			double tr = e.reward;

			if (!e.done) {
				size_t im = model(e.next).imax();

				tr += discount * target(e.next)[im];
			}

			Vec tQ_values = model(e.current);

			tQ_values[e.index] = tr;

			outs.push_back(tQ_values);
		}

		model.multithreaded_fit(ins, outs, 8);

		// Is it more sample efficient to replace the (updated)
		// experiences back into the buffer?
		for (auto &e : batch) {
			size_t im = model(e.next).imax();

			e.error = fabs(e.reward + discount
					* target(e.next)[im]
					- model(e.current)[e.index]);

			prbf.add(e);
		}
	}

	if (++frames > 25000) {
		frames = 0;

		target = model;
	}
}

}
