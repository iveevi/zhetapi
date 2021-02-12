# Colors
if(NOT WIN32)
	string(ASCII 27 esc)

	set(reset 	"${esc}[m")
	set(present	"${esc}[1;32m")
	set(notify	"${esc}[1;31m")
endif()