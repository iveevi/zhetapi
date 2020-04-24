#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>

#define MAX_COUNT	4096
#define FIFO_R_FILE	"input"
#define FIFO_W_FILE	"output"

using namespace std;

static void parent(void)
{
	uint8_t data[MAX_COUNT];
	uint8_t *in;
	size_t count;
	int fout;
	int fin;
	int n;

	fout = open(FIFO_W_FILE, O_WRONLY);
	fin = open(FIFO_R_FILE, O_RDONLY);

	assert(fout > 0);
	assert(fin > 0);

	do {
		printf("[Processor]: Waiting...\n");

		n = read(fin, &count, 4);

		printf("[Processor]: n [%d]\n", n);
		if (n != 4)
			break;

		in = new uint8_t(count);

		n = read(fin, in, count);
		assert(n == count);
		data[count] = 0;
		
		string s = "<" + string((char *) in) + ">\n";
		write(fout, s.c_str(), s.length() + 1);
	} while (count > 0);

	close(fout);
	close(fin);
}

int main()
{
	mknod(FIFO_R_FILE, S_IFIFO | 0666, 0);
	mknod(FIFO_W_FILE, S_IFIFO | 0666, 0);

	parent();

	printf("Exiting...\n");

	assert(!remove(FIFO_R_FILE));
	assert(!remove(FIFO_W_FILE));

	return 0;
}
