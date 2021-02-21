#include <vector>

#include <gnn.hpp>
#include <netnode.hpp>

using namespace std;
using namespace zhetapi::ml;

int main()
{
	NetNode nn1;
	NetNode nn2;
	NetNode nn3;
	NetNode nn4;

	nn1[1] << nn2[2];
	nn4[1] >> nn3[2];

	vector <NetNode <double> *> ins {&nn4, &nn2};

	cout << "gnn1:" << endl;
	GNN gnn1(ins);

	gnn1.trace();
	
	cout << "gnn2:" << endl;
	GNN gnn2(&nn4, &nn2);

	gnn2.trace();
}
