run:		tests
	@echo "\n[RUNNING PROGRAM]\n"
	./build/tests

debug:		tests
	@echo "\b[DEBUGGING PROGRAM]\n"
	gdb ./build/tests

graph:		build parsers
	@echo "[BUILDING GRAPHER]\n"
	nvcc web/graph.cpp -g -lfl -lGL -lglut -O3 -o build/graph

tests:		build parsers
	@echo "[BUILDING TESTS]\n"
	g++ tests/tests.cpp -lfl -lgmpxx -lgmp -g -o build/tests

ex:		build parsers
	@echo "[BUILDING TESTS]\n"
	g++ tests/ex.cpp -lfl -lgmpxx -lgmp -lglut -lGL -g -o build/ex

texifier:	build
	@echo "[BUILDING TEXIFIER]\n"
	
	touch build/texifier.in
	touch build/texifier.out
	
	g++ texifier/texifier.cpp -o build/texifier

driver:		build parsers
	@echo "[BUILDING DRIVER]\n"
	
	touch build/driver.in
	touch build/driver.out
	
	g++ web/driver.cpp -lfl -g -o build/driver -DDEBUG=0

cli:		build parsers
	@echo "[BUILDING CLI]\n"	
	g++ cli/cli.cpp -lfl -g -o build/cli -DDEBUG=0

parsers:
	flex -o build/lex.yy.c engine/lexer.l
	bison -t -o build/parser.tab.c engine/parser.y

build:
	@echo "[CREATING BUILD DIRECTORY]\n"
	mkdir build

all:		driver texifier tests graph
