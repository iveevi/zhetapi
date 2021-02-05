MODE="Release"
EXECUTABLE=""

if [ "$1" == "gdb" ]; then
	MODE="Debug"

	EXECUTABLE="gdb"
fi

cmake -DCMAKE_CXX_FLAGS=$MODE .
make rl

$EXECUTABLE ./rl