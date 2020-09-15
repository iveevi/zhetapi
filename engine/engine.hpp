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
	using rule = std::vector <std::pair <node_manager <T, U>, node_manager<T, U>>>;

	template <class T, class U>
	class Engine {
		rule <T, U> __std;
	public:
		// Load resources
		Engine(const std::string &);

		// Save resources
		~Engine();

		// Iterators to rules
		typename rule <T, U> ::iterator begin();
		typename rule <T, U> ::iterator end();

		const typename rule <T, U> ::const_iterator &begin() const;
		const typename rule <T, U> ::const_iterator &end() const;

		// Statics
		static std::vector <std::string > variables;

		// Friends
		friend class node_manager <T, U>;
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

			node_manager <T, U> f = node_manager <T, U> (first, variables);
			node_manager <T, U> s = node_manager <T, U> (second, variables);

			f.refactor_reference("u", new wildcard());

			__std.push_back({f, s});
		}
	}

	template <class T, class U>
	Engine <T, U> ::~Engine() {}

	// Iterators
	template <class T, class U>
	typename rule <T, U> ::iterator Engine <T, U> ::begin()
	{
		return __std.begin();
	}

	template <class T, class U>
	typename rule <T, U> ::iterator Engine <T, U> ::end()
	{
		return __std.end();
	}

	template <class T, class U>
	const typename rule <T, U> ::const_iterator &Engine <T, U> ::begin() const
	{
		return __std.begin();
	}

	template <class T, class U>
	const typename rule <T, U> ::const_iterator &Engine <T, U> ::end() const
	{
		return __std.end();
	}

}

#endif
