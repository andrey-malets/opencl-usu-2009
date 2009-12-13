#ifndef _LIBRARY_H	
#define _LIBRARY_H 1

#include <CL/cl.h>
#include <CImg/CImg.h>

#include <iostream>
#include <vector>

namespace opencl_usu_2009
{
	typedef unsigned char byte;

	template<typename Pixel = byte> class Identificator
	{
	public:
		/* Create an image in the device memory from CImg image source */
		Identificator(const cimg_library::CImg<Pixel>& source);

		/* Create an image in the device memory from the pixel vector */
		Identificator(const Pixel source[], const size_t width, const size_t height);

		/* Create empty image in the device memory of specified dimensions and default value */
		Identificator(const size_t width, const size_t height, Pixel value = Pixel());

		/* Create an image copying it's contents from the specified image */
		Identificator(const Identificator<Pixel> &other);

		/* Return image width */
		size_t width() const;

		/* Return image height */
		size_t height() const;

		/* Set the interest rectangle. All other operations will perform using this interest rectangle */
		void setInterestRect(const size_t x0, const size_t y0, const size_t width, const size_t height);

		/* Reset the interest rectangle to the full image */
		inline void clearInterestRect() { setInterestRect(0, 0, width(), height()); }

		/* Unload the image from device memory to the buffer at dest */
		void unload(Pixel *dest) const;

		/* Apply threshold processing to the interest rectangle */
		void trheshold(const Pixel value, const Pixel lessValue, const Pixel moreValue);

		/* Make the linear combination of interest rectangles of the current image and supplied image which
			in this case must be of the same dimensions, exception is thrown otherwise */
		void linearCombination(const Identificator<Pixel> other, const float a, const float b);

		/* Make the gauss filtration of the interest rectangle */
		void gauss(const float sigma, const size_t n);
	};
}

#endif // _LIBRARY_H
