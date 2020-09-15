# Tests
test:		test_build
	@echo "\n[RUNNING PROGRAM]\n"
	./build/tests < ./tests/tests.in

test_profile:	test_build_profile
	@echo "\n[RUNNING PROGRAM]\n"
	./build/tests < ./tests/tests.in

test_debug_exc:
	@echo "\b[DEBUGGING PROGRAM]\n"
	gdb ./build/tests

test_debug:	test_build_debug
	@echo "\b[DEBUGGING PROGRAM]\n"
	gdb ./build/tests

test_mem:	test_build
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes build/tests < tests/tests.in 

test_build:	inc/hidden	\
		build		\
		tests
	@echo "[BUILDING TESTS]\n"
	g++ -I engine -I inc/hidden -I inc/std -I build tests/tests.cpp -o build/tests

test_build_debug:	inc/hidden	\
			build		\
			tests
	@echo "[BUILDING TESTS]\n"
	g++ -I engine -I inc/hidden -I inc/std -I build tests/tests.cpp -g -o build/tests

test_build_profile:	inc/hidden	\
			build		\
			tests
	@echo "[BUILDING TESTS]\n"
	g++ -I engine -I inc/hidden -I inc/std -I build tests/tests.cpp -g -pg -o build/tests

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
		build		\
		tests
	@echo "[BUILDING ML TESTER]\n"
	g++ -I engine -I inc/hidden -I inc/std tests/ml.cpp -g -o build/ml

# GPU testing
gpu:		gpu_build
	@echo "\n[RUNNING GPU TESTS]\n"
	./build/gpu

gpu_build:	inc/gpu	\
		build	\
		tests
	@echo "[BUILDING GPU TESTER]\n"
	nvcc -I engine -I engine/hidden tests/gpu.cu -g -o build/gpu

# Manager testing
manager:	manager_build
	@echo "\n[RUNNING MANAGER]\n"
	./build/manager

manager_build:	mgr	\
		build
	@echo "[BUILDING MANAGER TESTER]\n"
	g++ mgr/config_manager.cpp -lboost_system -lboost_filesystem -o build/manager

# Build directory
build:
	mkdir build
