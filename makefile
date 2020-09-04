# Tests
test:		test_build
	@echo "\n[RUNNING PROGRAM]\n"
	./build/tests < ./tests/tests.in

test_debug:	test_build
	@echo "\b[DEBUGGING PROGRAM]\n"
	gdb ./build/tests

test_mem:	test_build
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes build/tests < tests/tests.in 

test_build:	inc/hidden	\
		tests
	@echo "[BUILDING TESTS]\n"
	g++ -I engine -I inc/hidden -I build tests/tests.cpp -lfl -lgmpxx -lgmp -g -o build/tests

# Machine learning
ml:		ml_build
	@echo "\n[RUNNING ML]\n"
	./build/ml

ml_debug:	ml_build
	@echo "\n[DEBUGGING ML]\n"
	gdb ./build/ml

ml_mem:		ml_build
	@echo "\n[DEBUGGING ML]\n"
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./build/ml

ml_build:	engine		\
		inc/std		\
		tests
	@echo "[BUILDING ML TESTER]\n"
	g++ -I engine -I inc/hidden -I inc/std tests/ml.cpp -g -lfl -o build/ml

# GPU testing
gpu:		gpu_build
	@echo "\n[RUNNING GPU TESTS]\n"
	./build/gpu

gpu_build:	inc/gpu	\
		tests
	@echo "[BUILDING GPU TESTER]\n"
	nvcc -I engine -I engine/hidden tests/gpu.cu -g -lfl -o build/gpu
