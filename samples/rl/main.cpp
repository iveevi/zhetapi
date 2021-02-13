#include "global.hpp"

// "Frame time"
const double delta(1.0/120.0);
const chrono::milliseconds frame((int) (1000 * delta));

// Agent
Agent agent;

// Networks

// On-line model
ml::NeuralNetwork <double> model(6, {
	ml::Layer <double> (4, new ml::ReLU <double> ()),
	ml::Layer <double> (8, new ml::Sigmoid <double> ()),
	ml::Layer <double> (4, new ml::ReLU <double> ())
});

// Confidence model (percentage error estimator)
ml::NeuralNetwork <double> confidence(6, {
	ml::Layer <double> (4, new ml::ReLU <double> ()),
	ml::Layer <double> (8, new ml::ReLU <double> ()),
	ml::Layer <double> (4, new ml::Sigmoid <double> ())
});

// Step through each iteration
void step()
{
	static double reward = 0;

	// Action
	double angle = 0;

	Vec P = agent.position;
	Vec V = agent.velocity;

	// Get the state
	Vec S = concat(P, V, F(P));

	double r = agent.runit();

	if (r > 0.5)
		angle = dirs[model(S).imax()];
	else
		angle = heurestic(P);

	Vec A = Vec::rarg(agent.force, angle);
	
	// Field
	Vec E = F(P);

	agent.move(A + E, delta);

	// Set angle
	agent.applied = A;
	agent.net = A + E;

	reward += agent.reward();
	if (!agent.in_bounds()) {
		agent.reset();

		cout << "Final reward = " << reward << endl;

		reward = 0;
	}
}

// Main function
int main(int argc, char **argv)
{
	// Remove std requirement for sin and cos
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

	return 0;
	for (size_t i = 0; i < 3600; i++) {

		this_thread::sleep_for(frame);
	}
}
