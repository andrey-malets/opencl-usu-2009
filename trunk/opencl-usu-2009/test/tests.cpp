#include "library.h"
#include <iostream>
#include <ctime>
#include <vector>
#include "CImg/CImg.h"
#include "../CImg/CPU_func.h"

void fillRandomBytes(byte*, size_t);
void average(std::vector<double>, double*, double*);
cimg_library::CImg<byte>* byteToCImg(byte[], size_t, size_t);
void printAverage(std::vector<std::vector<double> >);

void checkThreshold();
void checkLinear();
void checkGauss();

struct size
{
	size_t width;
	size_t height;
};

struct gp
{
	size_t s;
	size_t r;
};

const size_t repeateCount = 1;
const size_t executeCount = 1;
// Array of dimentions
const size d[] = {{320, 240}, {640, 480}, {1024, 768}, {1280, 1024}};//, {1980, 1080}, {3000, 2000}, {4000, 3000}};//, {7000, 7000}};
//const size d[] = {{2000, 2000}};
const std::vector<size> dimentions(d, d + sizeof(d) / sizeof(size));
std::vector<std::vector<double> > values(sizeof(d) / sizeof(size), std::vector<double>(repeateCount, 0));
const gp g[] = {{1, 3}, {2, 6}, {3, 9}, {4, 12}};
const std::vector<gp> gauss(g, g + sizeof(g) / sizeof(gp));

void thresholdOpencl()
{
	clock_t c;
	double t;

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
			for (int y = 0; y != executeCount; ++ y)
				id.trheshold(rand() % 256, rand() % 256, rand() % 256);
			id.unload(img);
			t = (double) (clock() - c) / CLOCKS_PER_SEC;
			values[i][k] = t;
		}
		// Delete image
		delete[] img;
	}
}

void linearCombinationOpencl()
{
	clock_t c;
	double t;

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
			opencl_usu_2009::ByteID id2(img, dimentions[i].width, dimentions[i].height);
			for (int y = 0; y != executeCount; ++ y)
				id.linearCombination(id2, 0.5, 0.5);
			id.unload(img);
			t = (double) (clock() - c) / CLOCKS_PER_SEC;
			values[i][k] = t;
		}
		// Delete image
		delete[] img;
	}
}

void gaussOpencl(double sigma, unsigned char radius)
{
	clock_t c;
	double t;

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
			opencl_usu_2009::ByteID id2(img, dimentions[i].width - 2 * radius, dimentions[i].height - 2 * radius);
			for (int y = 0; y != executeCount; ++ y)
				id.gauss(id2, sigma, radius);
			id.unload(img);
			t = (double) (clock() - c) / CLOCKS_PER_SEC;
			values[i][k] = t;
		}
		// Delete image
		delete[] img;
	}
}


void thresholdCpu()
{
	clock_t c;
	double t;

	for (int i = 0; i != dimentions.size(); ++i)
	{
		// Create image
		size_t size = dimentions[i].height * dimentions[i].width; // image size
		byte *img = new byte[size];
		fillRandomBytes(img, size);
		cimg_library::CImg<byte> *image = byteToCImg(img, dimentions[i].width, dimentions[i].height);

		for (int k = 0; k != repeateCount; ++k)
		{
			// Timing
			c = clock();
			for (int y = 0; y != executeCount; ++ y)
				porog(*image, *image, rand() % 256, rand() % 256, rand() % 256);
			t = (double) (clock() - c) / CLOCKS_PER_SEC;
			values[i][k] = t;
		}
		// Delete image
		delete image;
		delete[] img;
	}

}

void linearCombinationCpu()
{
	clock_t c;
	double t;

	for (int i = 0; i != dimentions.size(); ++i)
	{
		// Create image
		size_t size = dimentions[i].height * dimentions[i].width; // image size
		byte *img = new byte[size], *img2 = new byte[size];
		fillRandomBytes(img, size);
		fillRandomBytes(img2, size);
		cimg_library::CImg<byte> *image = byteToCImg(img, dimentions[i].width, dimentions[i].height);
		cimg_library::CImg<byte> *image2 = byteToCImg(img2, dimentions[i].width, dimentions[i].height);

		for (int k = 0; k != repeateCount; ++k)
		{
			// Timing
			c = clock();
			for (int y = 0; y != executeCount; ++ y)
				LineComb(*image, *image2, *image, 0.5, 0.5);
			t = (double) (clock() - c) / CLOCKS_PER_SEC;
			values[i][k] = t;
		}
		// Delete image
		delete image;
		delete[] img, img2;
	}

}

void gaussCpu(double sigma, unsigned char radius)
{
	clock_t c;
	double t;

	for (int i = 0; i != dimentions.size(); ++i)
	{
		// Create image
		size_t size = dimentions[i].height * dimentions[i].width; // image size
		byte *img = new byte[size];
		fillRandomBytes(img, size);
		cimg_library::CImg<byte> *image = byteToCImg(img, dimentions[i].width, dimentions[i].height);

		for (int k = 0; k != repeateCount; ++k)
		{
			// Timing
			c = clock();
			for (int y = 0; y != executeCount; ++ y)
					GaussBlur(*image, sigma, radius);
			t = (double) (clock() - c) / CLOCKS_PER_SEC;
			values[i][k] = t;
		}
		// Delete image
		delete image;
		delete[] img;
	}
}


int tests()
{

	byte b[100];
	opencl_usu_2009::ByteID f(b, 10, 10);
/*
	checkThreshold();
	return 1;

	checkLinear();
	return 1;

	checkGauss();
	return 1;
*/
	srand(clock());
/*
	// Treshold OpenCL
	std::cout << "Threshold OpenCL" << std::endl;
	thresholdOpencl();
	printAverage(values);

	// Treshold CPU
	std::cout << "Threshold CPU" << std::endl;
	thresholdCpu();
	printAverage(values);

	// LinearCombination OpenCL
	std::cout << "LinearCombination OpenCL" << std::endl;
	linearCombinationOpencl();
	printAverage(values);

	// LinearCombination CPU
	std::cout << "LinearCombination CPU" << std::endl;
	linearCombinationCpu();
	printAverage(values);
*/


	// Gauss OpenCL
	std::cout << "Gauss OpenCL" << std::endl;
	for ( int i = 0; i != gauss.size(); ++ i)
	{
		std::cout << gauss[i].s << " " << gauss[i].r << std::endl;
		gaussOpencl(gauss[i].s, gauss[i].r);
		printAverage(values);
	}

	// Gauss CPU
	std::cout << "Gauss CPU" << std::endl;
	for ( int i = 0; i != gauss.size(); ++ i)
	{
		std::cout << gauss[i].s << " " << gauss[i].r << std::endl;
		gaussCpu(gauss[i].s, gauss[i].r);
		printAverage(values);
	}
	
	return 1;
}

void checkThreshold()
{
	cimg_library::CImg<byte> image("3.bmp");
	size_t size = image.width() * image.height();
	byte *img = new byte[size];
	memcpy(img, image.data(), size);
	opencl_usu_2009::ByteID id(img, image.width(), image.height());
	//id.setInterestRect(220, 220, 240, 240);
	id.trheshold(0, 0, 255);
	id.unload(img);
	memcpy(image.data(), img, size);
	memcpy(image.data() + size, img, size);
	memcpy(image.data() + 2 * size, img, size);
	image.display();
	image.save("4.bmp");
	delete[] img;
}

void checkLinear()
{
	cimg_library::CImg<byte> image1("6.bmp");
	cimg_library::CImg<byte> image2("7.bmp");
	size_t size1 = image1.width() * image1.height();
	size_t size2 = image2.width() * image2.height();
	byte *img1 = new byte[size1];
	byte *img2 = new byte[size2];
	memcpy(img1, image1.data(), size1);
	memcpy(img2, image2.data(), size2);
	opencl_usu_2009::ByteID id1(img1, image1.width(), image1.height());
	opencl_usu_2009::ByteID id2(img2, image2.width(), image2.height());
	id1.setInterestRect(270, 120, 140, 124);
	id2.setInterestRect(120, 20, 140, 124);
	id1.linearCombination(id2, 0.5, -0.1);
	id1.unload(img1);
	memcpy(image1.data(), img1, size1);
	memcpy(image1.data() + size1, img1, size1);
	memcpy(image1.data() + 2 * size1, img1, size1);
	image1.display();
	image1.save("4.bmp");
	delete[] img1, img2;
}


void checkGauss()
{
	cimg_library::CImg<byte> image1("3.bmp");
	cimg_library::CImg<byte> image2(image1.width() - 80, image1.height() - 80, 1, 3);
	size_t size1 = image1.width() * image1.height();
	size_t size2 = image2.width() * image2.height();
	byte *img1 = new byte[size1];
	byte *img2 = new byte[size2];
	memcpy(img1, image1.data(), size1);
	memcpy(img2, image2.data(), size2);
	opencl_usu_2009::ByteID id1(img1, image1.width(), image1.height());
	opencl_usu_2009::ByteID id2(img2, image2.width(), image2.height());
	//id1.setInterestRect(270, 120, 140, 124);
	//id2.setInterestRect(120, 20, 140, 124);
	id1.gauss(id2, 13, 40);
	id2.unload(img2);
	memcpy(image2.data(), img2, size2);
	memcpy(image2.data() + size2, img2, size2);
	memcpy(image2.data() + 2 * size2, img2, size2);
	image2.display();
	image2.save("4.bmp");
	delete[] img1, img2;
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

cimg_library::CImg<byte>* byteToCImg(byte buffer[], size_t width, size_t height)
{
	size_t size = width * height;
	cimg_library::CImg<byte> *img = new cimg_library::CImg<byte>(width, height, 1, 3);
	memcpy(img->data(), buffer, size);
	memcpy(img->data() + size, buffer, size);
	memcpy(img->data() + 2 * size, buffer, size);
	return img;
}

void printAverage(std::vector<std::vector<double> > values)
{
	for (int i = 0; i != values.size(); ++i)
	{
		double m, d;
		average(values[i], &m, &d);
		std::cout << m << " " << d << std::endl;
	}
}
