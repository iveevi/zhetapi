graph: tests
	@echo "\n[BUILDING GRAPHER]\n"
	nvcc engine/graph.cpp -g -lfl -lGL -lglut -o build/graph

tests: texifier
	@echo "\n[BUILDING TESTS]\n"
	g++ engine/tests.cpp -lfl -lgmpxx -lgmp -g -o build/tests

texifier: driver
	@echo "\n[BUILDING TEXIFIER]\n"
	
	touch build/texifier.in
	touch build/texifier.out
	
	flex -o build/texifier.yy.c texifier/texifier.l
	bison -o build/texifier.tab.c texifier/texifier.y
	
	g++ texifier/texifier.cpp -lfl -o build/texifier -DDEBUG=0

driver: build
	@echo "\n[BUILDING DRIVER]\n"
	touch build/driver.in
	touch build/driver.out
	flex -o build/lex.yy.c engine/lexer.l
	bison -o build/parser.tab.c engine/parser.y
	g++ engine/driver.cpp -lfl -o build/driver -DDEBUG=0

build:
	@echo "\n[CREATING BUILD DIRECTORY]\n"
	mkdir build
