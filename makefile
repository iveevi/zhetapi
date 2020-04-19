tests: build texifier
	@echo "\n[BUILDING TESTS]\n"
	g++ engine/tests.cpp -lfl -g -o build/tests

texifier: build driver
	@echo "\n[BUILDING TEXIFIER]\n"
	touch build/texifier.in
	touch build/texifier.out
	flex texifier/lexer.l
	mv lex.yy.c build/
	bison texifier/parser.y
	mv parser.tab.c build/
	g++ texifier/texifier.cpp -lfl -o build/texifier -DDEBUG=0

driver: build
	@echo "\n[BUILDING DRIVER]\n"
	touch build/driver.in
	touch build/driver.out
	g++ engine/driver.cpp -lfl -o build/driver -DDEBUG=0

build:
	@echo "\n[CREATING BUILD DIRECTORY]\n"
	mkdir build
