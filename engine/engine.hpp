#ifndef ENGINE_H_
#define ENGINE_H_

// C/C++ headers
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>

namespace zhetapi {

	class Engine {
		std::unordered_map <int, std::vector <std::string>>	__simplifications;
		std::map <int, std::string>				__simplification_frequency;
	public:
		// Load resources
		Engine(const std::string &);

		// Save resources
		~Engine();
	};

	Engine::Engine(const std::string &path)
	{
		std::ifstream fin(path);

		using namespace std;

		std::string line;
		while (std::getline(fin, line)) {
			cout << "line: " << line << endl;
		}
	}

	Engine::~Engine() {}

}

#endif