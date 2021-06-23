#include "../engine/core/node_manager.hpp"
#include "../engine/core/enode.hpp"
#include "../engine/core/engine_base.hpp"
#include "../engine/engine.hpp"
#include "../engine/timer.hpp"
#include "../engine/core/enode_parser.hpp"

using namespace std;
using namespace zhetapi;

// Bench program
const size_t iters = 10000;

const std::string str = "64564 * 543 / 43432 + 54 + 423434 - 331232 * 4324234";

int main()
{
	Timer timer;

	// Etrees (new name for node_manager)
	Engine *ctx = new Engine();

	timer.start();
	for (size_t i = 0; i < iters; i++) {
		node_manager etree1(ctx, str);

		etree1.value(ctx);
	}
	timer.stop();

	long double t1 = timer.dt();

	cout << "node_manager time = " << t1/1000.0 << "ms." << endl;
	
	timer.start();
	for (size_t i = 0; i < iters; i++) {
		enode_value(strmake(str));
	}
	timer.stop();

	long double t2 = timer.dt();

	cout << "enode (undercut) time = " << t2/1000.0 << "ms." << endl;

	cout << "ratio: " << t1/t2 << endl;

	cout << "Simple test:" << endl;

	Enode en = strmake("5.0 * 7.0 / 320.0");
	en.print();

	Variant v = enode_value(en);

	cout << "v = " << v.str() << endl;
}
