#include <chrono>
#include <vector>
#include <unordered_map>

// Engine headers
#include <engine.hpp>

#include <core/rvalue.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

// Typedefs
using hrclk = chrono::high_resolution_clock;

// Timers
hrclk::time_point start_t;
hrclk::time_point end_t;

const int ITERS = 10000000;

// New struct
struct Scope {
        // Memory map (virtual "MMU")
        unordered_map <string, size_t> mmu;

        // Virtualized "memory"
        vector <Token *> mem;

        // Upper scope
        // Scope *next;

        void put(const string &str, Token *tptr) {
                if (mmu.find(str) == mmu.end()) {
                        // not yet mapped
                        mmu[str] = mem.size();

                        mem.push_back(tptr);
                } else {
                        // Consider deleting old stuff
                        mem[mmu[str]] = tptr;
                }
        }

        Token *operator[](size_t i) const {
                return mem[i];
        }
};

struct xvalue {
        string sym;
        size_t cin;     // Cached index
        size_t off;     // Scope offset

        xvalue(const string &str, Scope *scope) : sym(str) {
                // Ignore null scopes for now
                cin = scope->mmu[str];
        }

        Token *get(Scope *scope) const {
                return scope->mem[cin];
        }
};

int main()
{
        Engine *engine = new Engine();

        engine->put("x", new Operand <int> (10));
        engine->put("squared", new Operand <int> (10));
        engine->put("foo", new Operand <int> (10));
        engine->put("bar", new Operand <int> (10));

        start_t = hrclk::now();

        rvalue rv("squared");
        for (size_t i = 0; i < ITERS; i++)
                rv.get(engine);
        
        end_t = hrclk::now();

        cout << "Current time is: " << (chrono::duration_cast <chrono::milliseconds> (end_t - start_t)).count() << " ms" << endl;

        Scope *scope = new Scope;

        scope->put("x", new Operand <int> (10));
        scope->put("squared", new Operand <int> (10));
        scope->put("foo", new Operand <int> (10));
        scope->put("bar", new Operand <int> (10));

        start_t = hrclk::now();

        xvalue xv("squared", scope);
        for (size_t i = 0; i < ITERS; i++)
                xv.get(scope);
        
        end_t = hrclk::now();

        cout << "New time is: " << (chrono::duration_cast <chrono::milliseconds> (end_t - start_t)).count() << " ms" << endl;

	// const size_t id = zhp_id <OpZ> ();

	cout << "ID OF OP Int = " << zhp_id <OpZ> () << endl;
	cout << "ID OF OP Rat = " << zhp_id <OpQ> () << endl;
	cout << "ID OF OP Real = " << zhp_id <OpR> () << endl;
}