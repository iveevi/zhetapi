#include <iostream>

using namespace std;

int main()
{
	int block = 0;

	char c;

	string global;

	string b1;

	while (cin.get(c)) {
		cout << "c: " << c << endl;

		if (c == '\t')
			continue;

		if (c == '{') {
			block++;
			cout << "\tBlock Enter" << endl;
		}

		if (!block) {
			if (c == '\n')
				c = ',';
			
			global += c;
		}
		
		if (c == '}') {
			block--;
			cout << "\tBlock Leave" << endl;
		}
	}

	cout << "global:\n" << global << endl;
}