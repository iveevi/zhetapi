graph: build
	@echo "[BUILDING GRAPHER]\n"
	nvcc engine/graph.cpp -g -lfl -lGL -lglut -O3 -o build/graph

tests: build
	@echo "[BUILDING TESTS]\n"
	g++ engine/tests.cpp -lfl -lgmpxx -lgmp -g -o build/tests

texifier: build
	@echo "[BUILDING TEXIFIER]\n"
	
	touch build/texifier.in
	touch build/texifier.out
	
	g++ texifier/texifier.cpp -o build/texifier

driver: build
	@echo "[BUILDING DRIVER]\n"
	touch build/driver.in
	touch build/driver.out
	flex -o build/lex.yy.c engine/lexer.l
	bison -o build/parser.tab.c engine/parser.y
	g++ engine/driver.cpp -lfl -g -o build/driver -DDEBUG=0

build:
	@echo "[CREATING BUILD DIRECTORY]\n"
	mkdir build

all: driver texifier tests graph
