#include "CImg.h"
using namespace cimg_library;

//1. пороговая обработка изображения k, с записью результата в изображение j 
//Ij[x,y] = (Ik[x,y]>=T)?a:b. T,a,b - параметры

void porog(const CImg<unsigned char>& input, CImg<unsigned char>& output, unsigned char t, unsigned char a, unsigned char b )
{
	int length = input.width() * input.height();
	int height = input.height();
	for (int i = 0; i < length; i++)
	{
		output.data()[i] = output.data()[i+length] = output.data()[i+2*length] = (input.data()[i] >= t) ? a : b; 
	}
}
//2. линейная комбинация изображений
//Ij[x,y] = a*Ik[x,y] + b*Ii[x,y]

void LineComb(const CImg<unsigned char>& Ik, const CImg<unsigned char>& Ii, CImg<unsigned char>& Ij, float a, float b )
{
	int length = Ik.width() * Ik.height();
	for (int i = 0; i < length; i++)
	{
		float res = a * (Ik.data()[i]) + b * (Ii.data()[i]);
		res = (res > 255) ? 255 : res;
		res = (res < 0) ? 0 : res;
		Ij.data()[i] = Ij.data()[i+length] = Ij.data()[i+2*length] = (unsigned char)res; 
	}
}

//3. Гауссова фильтрация с параметрами sigma - параметр гауссиана, n - размер локального окна
//GB(x,y) = 1/(2*pi*sigma*sigma) * exp( - (x*x + y*y) / (2 * sigma*sigma))
// blur: true - размытие, false - повышние резкости
template<typename Data> void GaussBlur(CImg<Data>& input, double sigma, const unsigned char n, const bool blur = true)
{
#define M_PI       3.14159265358979323846

	int m = (blur) ? 1 : -1;

	double *mask = (double *)malloc(sizeof(double)*(2*n+1)*(2*n+1));
	double s = 0;
	for(int j = -n; j < n+1; j++)
		for (int i = -n; i < n+1; i++)
			s += mask[(i+n) + (j+n)*(2*n+1)] =  m /(2*M_PI*sigma*sigma) * exp( (double) - (i*i + j*j) / (2 * sigma*sigma));

	if (!blur)
		s = -s;
	for (int i = 0; i < (2*n+1)*(2*n+1); i++)
		mask[i] /= s;
	
	if (!blur)
		mask[2*(n+1)*n] += 2;

	CImg<Data> additioanl(input.width(), input.height(), 1, 3);
	additioanl.fill(0);

	int length = input.width() * input.height();
	int width = input.width();

	for (int i = 0; i < length; i++)
	{
		double newpix = 0;
		double si = 0;
		for(int j = -n; j < n+1; j++)
			for (int l = -n; l < n+1; l++)
			{
				int index = i + l + j*width;
				if ( index >= 0 && index < length)
					newpix += mask[(l+n) + (j+n)*(2*n+1)] * input.data()[index];
			}

		Data res = (Data) newpix;
		if (s > 255)
			res = 255;
		if (s < 0)
			res = 0;
		additioanl.data()[i] = additioanl.data()[i+length] = additioanl.data()[i+2*length] = res; 
	}	

	memcpy(input.data(),additioanl.data(), 3*length);

	free(mask);
}