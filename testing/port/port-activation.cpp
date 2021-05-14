#include "port.hpp"

// TODO: add another for double activations (take account for error)
static bool act_general(ostringstream &oss,
		const std::string &prefix,
		const vector <zhetapi::ml::Activation <int> *> &acts,
		const vector <zhetapi::Vector <int>> &ins,
		const vector <vector <zhetapi::Vector <int>>> &outs,
		const vector <vector <zhetapi::Vector <int>>> &douts)
{
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

			if (acts[i]->compute(ins[j]) != outs[i][j])
				return false;
		}
	}

	vector <zhetapi::ml::Activation <int> *> dacts;
	for (auto act : acts)
		dacts.push_back(act->derivative());
	
	for (size_t i = 0; i < dacts.size(); i++) {
		oss << endl;
		oss << "Next activation derivative:" << endl;
		for (size_t j = 0; j < ins.size(); j++) {
			oss << prefix << (i + 1) << "(input #" << (j + 1)
				<< ") = " << dacts[i]->compute(ins[j]) << endl;
			oss << "should equal " << douts[i][j] << endl;

			if (dacts[i]->compute(ins[j]) != douts[i][j])
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
	using namespace zhetapi;
	using namespace zhetapi::ml;

	return act_general(oss,
		"linear",
		{
			new Linear <int> (),
			new Linear <int> (2)
		},
		{
			Vector <int> {1, 2, 3, 4}
		},
		{
			{Vector <int> {1, 2, 3, 4}},
			{Vector <int> {2, 4, 6, 8}}
		},
		{
			{Vector <int> {1, 1, 1, 1}},
			{Vector <int> {2, 2, 2, 2}}
		});
}

TEST(act_relu)
{
	using namespace zhetapi;
	using namespace zhetapi::ml;

	return act_general(oss,
		"relu",
		{
			new ReLU <int> ()
		},
		{
			Vector <int> {1, 2, 3, 4},
			Vector <int> {1, -1, 3, -1}
		},
		{
			{
				Vector <int> {1, 2, 3, 4},
				Vector <int> {1, 0, 3, 0}
			}
		},
		{
			{
				Vector <int> {1, 1, 1, 1},
				Vector <int> {1, 0, 1, 0}
			}
		});
}