#include "global.hpp"

// "Frame time"
const double delta(1.0/120.0);
const chrono::milliseconds frame((int) (1000 * delta));

// Agent
Agent agent(0.95);

// Replay buffer
replays prbf(1000, 500);

// Step through each iteration
void step()
{
	static double reward = 0;
	static double error = 0;
	static size_t frames = 0;

	// Experience
	experience e;

	// Action
	double angle = 0;

	Vec P = agent.position;
	Vec V = agent.velocity;

	// Get the state
	Vec S = agent.state();

	e.current = S;

	// Q-value
	Vec Q_values = model(S);

	// Get the action
	double rnd = agent.runit();

	int i = -1;
	if (rnd > 0.1) {
		i = Q_values.imax();

		e.index = i;

		angle = dirs[i];
	} else {
		angle = heurestic(P);

		i = 2 * (angle/(acos(-1)) - 0.5);

		e.index = i;
	}

	// Create the action force
	Vec A = Vec::rarg(agent.force, angle);
	
	// Field
	agent.move(A + F(P), delta);

	// Next state
	Vec N = agent.state();

	e.next = N;

	// Reward
	double r = agent.reward();

	e.reward = r;

	// TD-error
	double err = fabs(r + agent.gamma * model(N).max() - Q_values[i]);

	e.error = err;

	// Update static values
	reward += r;
	error += err;
	frames++;

	// Decide termination
	if (!agent.in_bounds()) {
		agent.reset();

		cout << "Final reward = " << reward
			<< "\tframes last = " << frames
			<< "\taverage TD-error = "
			<< error/frames << endl;

		e.done = true;

		reward = 0;
		error = 0;
		frames = 0;
	}

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

			if (!e.done)
				tr += agent.gamma * model(e.next).max();

			Vec tQ_values = model(e.current);

			tQ_values[e.index] = tr;

			outs.push_back(tQ_values);
		}

		model.multithreaded_fit(ins, outs, 8);

		// Is it more sample efficient to replace the (updated)
		// experiences back into the buffer?
		for (auto &e : batch) {
			e.error = fabs(e.reward + agent.gamma
					* model(e.next).max()
					- model(e.current)[e.index]);

			prbf.add(e);
		}
	}
}

// Main function
int main(int argc, char **argv)
{
	// Initialize the model (refactor erfs to do without the 'error' part)
	ml::Optimizer <double> *opt = new ml::Adam <double> ();
	ml::Erf <double> *cost = new ml::MeanSquaredError <double> ();

	model.set_optimizer(opt);
	model.set_cost(cost);

	// Glut initialization
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA
			| GLUT_DEPTH
			| GLUT_DOUBLE
			| GLUT_MULTISAMPLE);

	glEnable(GL_MULTISAMPLE);

	glutInitWindowSize(640, 480);
	glutCreateWindow("Reinforcement Learning Simulation");

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutTimerFunc(1000 * delta, timer, 1);

	glutMainLoop();

	// Free resources
	delete opt;
	delete cost;

	return 0;
}
