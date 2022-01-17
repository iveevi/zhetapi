#include "global.hpp"

using namespace zhetapi;
using namespace zhetapi::ml;

static bool act_general(ostream &oss,
		const std::string &prefix,
		const vector <Activation <double> *> &acts,
		const vector <Vector <double>> &ins,
		const vector <vector <Vector <double>>> &outs)
{
	static double epsilon = 1e-10;

	oss << "Inputs:" << endl;
	for (auto v : ins)
		oss << "\t" << v << endl;
	
	for (size_t i = 0; i < acts.size(); i++) {
		oss << endl;
		oss << "Next activation:" << endl;
		for (size_t j = 0; j < ins.size(); j++) {
			oss << prefix << (i + 1) << "(input #" << (j + 1)
				<< ") = " << acts[i]->compute(ins[j]) << endl;
			oss << "should equal " << outs[i][j] << endl;
			oss << "diff = " << (acts[i]->compute(ins[j]) - outs[i][j]).norm() << endl;

			if ((acts[i]->compute(ins[j]) - outs[i][j]).norm() > epsilon)
				return false;
		}
	}

	vector <Activation <double> *> dacts;
	for (auto act : acts)
		dacts.push_back(act->derivative());
	
	for (size_t i = 0; i < dacts.size(); i++) {
		oss << endl;
		oss << "Next activation derivative:" << endl;
		for (size_t j = 0; j < ins.size(); j++) {
			Vector <double> dout(ins[j].size());

			// Use gradient checking
			for (size_t k = 0; k < ins[j].size(); k++) {
				Vector <double> back = ins[j];
				Vector <double> forward = ins[j];

				back[k] -= epsilon;
				forward[k] += epsilon;

				dout[k] = (acts[i]->compute(forward)
					- acts[i]->compute(back))[k]/(2 * epsilon);
			}

			oss << prefix << (i + 1) << "(input #" << (j + 1)
				<< ") = " << dacts[i]->compute(ins[j]) << endl;
			oss << "should equal " << dout << endl;
			oss << "diff = " << (dacts[i]->compute(ins[j]) - dout).norm() << endl;

			if ((dacts[i]->compute(ins[j]) - dout).norm() > 1e-5)
				return false;
		}
	}

	for (auto act : acts)
		delete act;

	for (auto dact : dacts)
		delete dact;

	return true;
}

TEST(act_linear)
{
	return act_general(oss,
		"linear",
		{
			new Linear <double> (),
			new Linear <double> (2)
		},
		{
			Vector <double> {1, 2, 3, 4}
		},
		{
			{Vector <double> {1, 2, 3, 4}},
			{Vector <double> {2, 4, 6, 8}}
		});
}

TEST(act_relu)
{
	return act_general(oss,
		"relu",
		{
			new ReLU <double> ()
		},
		{
			Vector <double> {1, 2, 3, 4},
			Vector <double> {1, -1, 3, -1}
		},
		{
			{
				Vector <double> {1, 2, 3, 4},
				Vector <double> {1, 0, 3, 0}
			}
		});
}

TEST(act_leaky_relu)
{
	return act_general(oss,
		"leaky relu",
		{
			new LeakyReLU <double> (0.2)
		},
		{
			Vector <double> {1, 2, 3, 4},
			Vector <double> {1, -1, 3, -2}
		},
		{
			{
				Vector <double> {1, 2, 3, 4},
				Vector <double> {1, -0.2, 3, -0.4}
			}
		});
}

TEST(act_sigmoid)
{
	return act_general(oss,
		"sigmoid",
		{
			new Sigmoid <double> ()
		},
		{
			Vector <double> {0.5, 2, 0, 4},
			Vector <double> {1, -1, 3, -2}
		},
		{
			{
				Vector <double> {
					0.622459331202,
					0.880797077978,
					0.5,
					0.982013790038},
				Vector <double> {
					0.73105857863,
					0.26894142137,
					0.952574126822,
					0.119202922022}
			}
		});
}
