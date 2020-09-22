# Portability
port:		port_build
	./build/port < tests/input > output.log
	-diff --color -w tests/output output.log
	rm output.log

port_build:	inc/hidden	\
		build		\
		tests
	g++ -I engine -I inc/hidden -I inc/std -I build tests/portability.cpp -o build/port

# Tests
test:		test_build
	@echo "\n[RUNNING PROGRAM]\n"
	./build/tests < ./tests/input

test_raw:
	./build/tests < ./tests/input

test_profile:	test_build_profile
	@echo "\n[PROFILING TEST PROGRAM]\n"
	./build/tests < ./tests/input
	gprof ./build/tests > prof.out

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
	g++ -I engine -I inc/hidden -I inc/std -I build tests/tests.cpp -ldl -o build/tests

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

manager_build:	build
	@echo "[BUILDING MANAGER TESTER]\n"
	g++ cfg/config_manager.cpp -lboost_system -lboost_filesystem -o build/manager

# CLI testing
cli:		cli_build
	@echo "\n[RUNNING CLI]\n"
	./build/cli

cli_debug:		cli_build_debug
	@echo "\n[RUNNING CLI]\n"
	gdb ./build/cli

cli_build:	build
	@echo "[BUILDING CLI TESTER]\n"
	g++ -I engine -I inc/std -I inc/hidden cli/cli.cpp -lboost_system -lboost_filesystem -o build/cli

cli_build_debug:	\
		build
	@echo "[BUILDING CLI TESTER]\n"
	g++ -I engine -I inc/std -I inc/hidden cli/cli.cpp -g -lboost_system -lboost_filesystem -o build/cli

# Physics engine testing
physics:	physics_build
	@echo "\n[RUNNING CLI]\n"
	./build/physics

physics_build:	build
	@echo "\n[RUNNING CLI]\n"
	g++ -I engine physics/line.cpp -lglut -lGL -lGLEW -lglfw -o build/physics

# Opengl testing
opengl:		opengl_build
	@echo "\n[RUNNING CLI]\n"
	./build/opengl

opengl_build:	build
	@echo "\n[RUNNING CLI]\n"
	g++ -I engine -I physics physics/main.cpp physics/shader.cpp physics/texture.cpp -lglut -lGL -lGLEW -lglfw -o build/opengl

opengl_debug:	opengl_build_debug
	@echo "\n[RUNNING CLI]\n"
	gdb ./build/opengl_debug

opengl_build_debug:	build
	@echo "\n[RUNNING CLI]\n"
	g++ -I engine -I physics -g physics/main.cpp -lglm -lglut -lGL -lGLEW -lglfw -o build/opengl_debug

# Build directory
build:
	mkdir build
