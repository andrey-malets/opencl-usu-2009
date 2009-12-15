#include "CImg.h"
#include <iostream>
#include <math.h>
#include "CPU_func.h"

using namespace cimg_library;

int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		std::cout<<"usage:\t CImg.exe <path_to_bmp>";
		return -1;
	}
	CImg<unsigned char> def(argv[1]);
	CImg<unsigned char> forGaus(argv[1]);

	CImg<unsigned char> result(forGaus.width(), forGaus.height(), 1, 3);

	forGaus.display();
	GaussBlur(forGaus, 2, 6, false);
	forGaus.display();

	LineComb(forGaus, def, result, 1, -1);
	result.display();

	porog(result, result, 5, 255, 0);
	result.display();

	GaussBlur(result, 4, 12);
	result.display();

	return 0;
}
