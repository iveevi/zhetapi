run:		tests
	@echo "\n[RUNNING PROGRAM]\n"
	./build/tests < ./tests/tests.in

debug:		tests
	@echo "\b[DEBUGGING PROGRAM]\n"
	gdb ./build/tests

ml:		ml_build
	@echo "\n[RUNNING PROGRAM]\n"
	./build/ml

ml_build:	build parsers
	@echo "[BUILDING GRAPHER]\n"
	g++ -I engine -I engine/hidden tests/ml.cpp -g -lfl -o build/ml

graph:		build parsers
	@echo "[BUILDING GRAPHER]\n"
	nvcc web/graph.cpp -g -lfl -lGL -lglut -O3 -o build/graph

tests:		build parsers
	@echo "[BUILDING TESTS]\n"
	g++ -I engine -I engine/hidden -I build tests/tests.cpp -lfl -lgmpxx -lgmp -g -o build/tests

exp:		build parsers
	@echo "[BUILDING TESTS]\n"
	g++ -I engine tests/exp.cpp -lfl -lgmpxx -lgmp -lglut -lGL -g -o build/exp
	./build/exp

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
	flex -o build/lex.yy.c engine/hidden/lexer.l
	bison -t -o build/parser.tab.c engine/hidden/parser.y

build:
	@echo "[CREATING BUILD DIRECTORY]\n"
	mkdir build

all:		driver texifier tests graph
