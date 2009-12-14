#include "library.h"
#include <iostream>
#include <ctime>
#include <vector>

void fillRandomBytes(byte*, size_t);
void average(std::vector<double>, double*, double*);

struct size
{
	size_t width;
	size_t height;
};

const size_t repeateCount = 10;
// Array of dimentions
//const size d[] = {{320, 240}, {640, 480}, {1024, 768}, {1280, 1024}, {1600, 1200}};
const size d[] = {{2000, 2000}};
const std::vector<size> dimentions(d, d + sizeof(d) / sizeof(size));

int tests()
{

	clock_t c;
	double t;
	std::vector<std::vector<double> > values(sizeof(d) / sizeof(size), std::vector<double>(repeateCount, 0));

	byte b[100];
	opencl_usu_2009::ByteID f(b, 10, 10);
	
	srand(clock());

	for (int i = 0; i != dimentions.size(); ++i)
	{
		// Create image
		size_t size = dimentions[i].height * dimentions[i].width; // image size
		byte *img = new byte[size];
		fillRandomBytes(img, size);

		for (int k = 0; k != repeateCount; ++k)
		{
			// Timing
			c = clock();
			opencl_usu_2009::ByteID id(img, dimentions[i].width, dimentions[i].height);
			id.trheshold(rand() % 256, rand() % 256, rand() % 256);
			id.unload(img);
			t = (double) (clock() - c) / CLOCKS_PER_SEC;
			values[i][k] = t;
		}
		// Delete image
		delete[] img;
	}

	for (int i = 0; i != values.size(); ++i)
	{
		double m, d;
		average(values[i], &m, &d);
		std::cout << m << " " << d << std::endl;
	}

	exit(0);
}

void fillRandomBytes(byte *buffer, size_t size)
{
	srand(clock());
	for(size_t i = 0; i != size; ++i)
		buffer[i] = rand() % 256;
}

void average(std::vector<double> vec, double *mm, double *dd)
{
	double m = 0, m2 = 0, d = 0;
	for (int i = 0; i != vec.size(); ++i)
	{
		m += vec[i];
		m2 += vec[i] * vec[i];
	}
	m /= vec.size();
	m2 /= vec.size();

	d = sqrt(m2 - m * m);

	*mm = m, *dd = d;
}
