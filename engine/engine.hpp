#ifndef ENGINE_H_
#define ENGINE_H_

// C/C++ headers
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

// Engine headers
#include <node_manager.hpp>

namespace zhetapi {

	template <class T, class U>
	class Engine {
		std::vector <std::pair <node_manager <T, U>, node_manager<T, U>>> __std;
	public:
		// Load resources
		Engine(const std::string &);

		// Save resources
		~Engine();

		static std::vector <std::string > variables;
	};

	template <class T, class U>
	std::vector <std::string> Engine <T, U> ::variables = {"u"};

	template <class T, class U>
	Engine <T, U> ::Engine(const std::string &path)
	{
		std::ifstream fin(path);

		using namespace std;

		std::string line;
		while (std::getline(fin, line)) {
			// Extract parts of the line
			auto itr = line.find(":");

			std::string first, second;

			first = line.substr(0, itr);
			second = line.substr(itr + 1);

			cout << "----------------------------------------------" << endl;
			cout << "line: " << line << endl;

			node_manager <T, U> f = node_manager <T, U> (first, variables);
			node_manager <T, U> s = node_manager <T, U> (second, variables);

			f.print();
			s.print();
		}
	}

	template <class T, class U>
	Engine <T, U> ::~Engine() {}

}

#endif