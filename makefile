# GPU testing
gpu:		gpu_build
	@echo "\n[RUNNING GPU TESTS]\n"
	./build/gpu

gpu_build:	build
	@echo "[BUILDING GPU TESTER]\n"
	nvcc -I engine -I engine/hidden tests/gpu.cu -g -lfl -o build/gpu

mem:		tests
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes build/tests < tests/tests.in 

run:
	@echo "[BUILDING TESTS]\n"
	g++ -I engine -I inc/hidden -I build tests/tests.cpp -lfl -lgmpxx -lgmp -g -o build/tests

	@echo "\n[RUNNING PROGRAM]\n"
	./build/tests < ./tests/tests.in

debug:		tests
	@echo "\b[DEBUGGING PROGRAM]\n"
	gdb ./build/tests

# Machine Learning
ml:		ml_build
	@echo "\n[RUNNING ML]\n"
	./build/ml

ml_debug:	ml_build
	@echo "\n[DEBUGGING ML]\n"
	gdb ./build/ml

ml_mem:		ml_build
	@echo "\n[DEBUGGING ML]\n"
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./build/ml

ml_build:	build
	@echo "[BUILDING ML TESTER]\n"
	g++ -I engine -I inc/hidden -I inc/std tests/ml.cpp -g -lfl -o build/ml

# Other
graph:		build
	@echo "[BUILDING GRAPHER]\n"
	nvcc web/graph.cpp -g -lfl -lGL -lglut -O3 -o build/graph

tests:		build
	@echo "[BUILDING TESTS]\n"
	g++ -I engine -I engine/hidden -I build tests/tests.cpp -lfl -lgmpxx -lgmp -g -o build/tests

exp:		build
	@echo "[BUILDING TESTS]\n"
	g++ -I engine tests/exp.cpp -lfl -lgmpxx -lgmp -lglut -lGL -g -o build/exp
	./build/exp

texifier:	build
	@echo "[BUILDING TEXIFIER]\n"
	
	touch build/texifier.in
	touch build/texifier.out
	
	g++ texifier/texifier.cpp -o build/texifier

driver:		build
	@echo "[BUILDING DRIVER]\n"
	
	touch build/driver.in
	touch build/driver.out
	
	g++ web/driver.cpp -lfl -g -o build/driver

cli:		cli_build
	@echo "\n[RUNNING CLI]\n"
	./build/cli

cli_build:	build
	@echo "[BUILDING CLI]\n"	
	g++ cli/cli.cpp -lfl -g -o build/cli

build:
	@echo "[CREATING BUILD DIRECTORY]\n"
	mkdir build

all:		driver texifier tests graph
