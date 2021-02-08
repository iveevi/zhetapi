#include <iostream>
#include <sstream>

#include <unistd.h>

// Curl library
#include <curl/curl.h>

using namespace std;

const string url = "https://query1.finance.yahoo.com/v7/finance/download/"; 

string stocks[] = {
	"GME",
	"^GSPC"
};

struct stock_stamp {
	double open;
	double high;
	double low;
	double close;
	double adj;	// Adjacent close
};

ostream &operator<<(ostream &os, const stock_stamp &s)
{
	os << "[close = " << s.close << ", open = " << s.open << "]";

	return os;
}

size_t write_to_stock(char *ptr, size_t size, size_t nmemb, stock_stamp *s)
{
	size_t count = size * nmemb;

	string str(ptr, count);

	istringstream iss(str);

	iss.ignore(256, '\n');
	iss.ignore(256, ',');	// Ignore date

	char c;

	iss >> s->open >> c
		>> s->high >> c
		>> s->low >> c
		>> s->close >> c
		>> s->adj >> c;
	
	return count;
}

struct load_stock_exception {};

stock_stamp load_stock(string identifier)
{
	stock_stamp ss;

	CURL *curl;

	curl = curl_easy_init();

	stock_stamp gme;
	if (curl) {
		curl_easy_setopt(curl, CURLOPT_URL, (url + identifier).c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_stock);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ss);

		if (curl_easy_perform(curl) != CURLE_OK)
			throw load_stock_exception();

		curl_easy_cleanup(curl);
	}

	return ss;
}

int main()
{
	for (size_t i = 0; i < 1000; i++) {
		printf("\x1B[2J\x1B[H");

		stock_stamp gme = load_stock(stocks[0]);

		cout << "gme: " << gme << endl;

		stock_stamp snp = load_stock(stocks[1]);

		cout << "snp: " << snp << endl;

		usleep(5e6);
	}
}
