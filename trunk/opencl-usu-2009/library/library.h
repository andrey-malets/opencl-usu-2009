#ifndef _LIBRARY_H	
#define _LIBRARY_H 1

#include "CImg/CImg.h"
#include "CL/cl.h"

#include <iostream>
#include <stdexcept>

/* throw specification ignored */
#pragma warning(disable: 4290)

namespace opencl_usu_2009
{
	typedef unsigned char byte;

	class LibraryException : public std::exception { };
	class DimensionException : public LibraryException { };
	class APIException : public LibraryException { };
	class OutOfMemoryException : public APIException { };

	class Common
	{
	public:
		/* Return image width */
		size_t getWidth() const throw() { return width; }

		/* Return image height */
		size_t getHeight() const throw() { return height; }

		/* Set the interest rectangle. All other operations will perform using this interest rectangle */
		void setInterestRect(const size_t x0, const size_t y0, const size_t width, const size_t height) throw (DimensionException);

		/* Reset the interest rectangle to the full image */
		inline void clearInterestRect() throw() { setInterestRect(0, 0, getWidth(), getHeight()); }

	protected:
		Common(size_t width, size_t height);
		Common(const Common &other);
		~Common();

		size_t getX() { return x; }
		size_t getY() { return y; }

		size_t getIRWidth() { return ir_width; }
		size_t getIRHeight() { return ir_height; }

		cl_command_queue getQueue() { return command_queue; }
		cl_context getContext() { return context; }

		static void check(cl_int);

	private:
		Common() { }

		static void init();
		static void finalize();

		static size_t refcount;
		static cl_context context;
		static cl_command_queue command_queue;
		size_t width, height, ir_width, ir_height, x, y;
	};

	template<typename Pixel = byte> class Identificator : public Common
	{
	public:
		/* Create an image in the device memory from CImg image source */
		Identificator(const cimg_library::CImg<Pixel>& source) throw(APIException);

		/* Create an image in the device memory from the pixel vector */
		Identificator(const Pixel *source, const size_t width, const size_t height) throw(APIException) : Common(width, height)
		{
			cl_int err;
			size_t size = sizeof(Pixel) * getWidth() * getHeight();
			buffer = clCreateBuffer(getContext(), CL_MEM_READ_WRITE, size, (void *)source, &err);
			check(err);

			err = clEnqueueWriteBuffer(getQueue(), buffer, CL_TRUE, 0, size, source, 0, NULL, NULL);
			try { check(err); }
			catch(...)
			{
				clReleaseMemObject(buffer);
				throw;
			}
		}

		/* Create empty image in the device memory of specified dimensions and default value */
		Identificator(const size_t width, const size_t height, Pixel value) throw(APIException);

		/* Create an image copying it's contents from the specified image */
		Identificator(const Identificator<Pixel> &other) throw(APIException)
			: Common(other.getWidth(), other.getHeight()), buffer(other.buffer)
		{
			clRetainMemObject(buffer);
		}

		/* Unload the image from device memory to the buffer at dest */
		void unload(Pixel *dest) const throw (APIException);

		/* Apply threshold processing to the interest rectangle */
		void trheshold(const Pixel value, const Pixel lessValue, const Pixel moreValue) throw (APIException);

		/* Make the linear combination of interest rectangles of the current image and supplied image which
		in this case must be of the same dimensions, exception is thrown otherwise */
		void linearCombination(const Identificator<Pixel> other, const float a, const float b) throw (APIException);

		/* Make the gauss filtration of the interest rectangle */
		void gauss(const float sigma, const size_t n) throw (APIException);

		/* Make a copy of this buffer in the device memory */
		Identificator copy()
		{
			return Identificator(buffer, *this);
		}

		~Identificator()
		{
			clReleaseMemObject(buffer);
		}

	private:
		Identificator operator=(const Identificator<Pixel> &rhs) { /* no, thanks */ }

		Identificator(const cl_mem otherBuffer, const Common &other) : buffer(buffer), Common(other)
		{
			cl_int err;
			size_t size = sizeof(Pixel) * getWidth() * getHeight();
			buffer = clCreateBuffer(getContext(), CL_MEM_READ_WRITE, size, NULL, &err);
			check(err);

			err = clEnqueueCopyBuffer(getQueue(), otherBuffer, buffer, 0, 0, size, 0, NULL, NULL);
			check(err);
		}

		cl_mem buffer;
	};
}

#endif // _LIBRARY_H
