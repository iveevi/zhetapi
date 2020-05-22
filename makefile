graph:		build parsers
	@echo "[BUILDING GRAPHER]\n"
	nvcc engine/graph.cpp -g -lfl -lGL -lglut -O3 -o build/graph

tests:		build parsers
	@echo "[BUILDING TESTS]\n"
	g++ engine/tests.cpp -lfl -lgmpxx -lgmp -g -o build/tests

texifier:	build
	@echo "[BUILDING TEXIFIER]\n"
	
	touch build/texifier.in
	touch build/texifier.out
	
	g++ texifier/texifier.cpp -o build/texifier

driver:		build parsers
	@echo "[BUILDING DRIVER]\n"
	
	touch build/driver.in
	touch build/driver.out
	
	g++ engine/driver.cpp -lfl -g -o build/driver -DDEBUG=0

parsers:
	flex -o build/lex.yy.c engine/lexer.l
	bison -o build/parser.tab.c engine/parser.y

build:
	@echo "[CREATING BUILD DIRECTORY]\n"
	mkdir build

all:		driver texifier tests graph
