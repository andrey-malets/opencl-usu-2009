#ifndef _LIBRARY_H	
#define _LIBRARY_H 1

#include "CImg/CImg.h"
#include "CL/cl.h"

#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

/* throw specification ignored */
#pragma warning(disable: 4290)

namespace opencl_usu_2009
{
	typedef unsigned char byte;

	class LibraryException : public std::exception { };

	class DimensionException : public LibraryException
	{
	public:
		DimensionException(size_t expectedWidth, size_t expectedHeight, size_t actualWidth, size_t actualHeight)
			: expectedWidth(expectedWidth),
			  expectedHeight(expectedHeight),
			  actualWidth(actualWidth),
			  actualHeight(actualHeight)
		{}

		size_t getExpectedWidth() { return expectedWidth; }
		size_t getExpectedHeight() { return expectedHeight; }

		size_t getActualWidth() { return actualWidth; }
		size_t getActualHeight() { return actualHeight; }

	private:
		size_t expectedWidth, expectedHeight, actualWidth, actualHeight;
	};

	class APIException : public LibraryException
	{
	public:
		APIException(cl_int code): errorCode(code) { }
		int getErrorCode() { return errorCode; }

	private:
		cl_int errorCode;
	};

	class OutOfMemoryException : public APIException { };

	class Common
	{
	public:

		/* File with kernels source code */
		static const char *kernelsFile;

		/* OpenCL compiler build options */
		static const char *buildOptions;

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

		size_t getX() const { return x; }
		size_t getY() const { return y; }

		size_t getIRWidth() const { return ir_width; }
		size_t getIRHeight() const { return ir_height; }

		static cl_command_queue getQueue() { return command_queue; }
		static cl_context getContext() { return context; }
		static cl_program getProgram() { return program; }

		void execute(cl_kernel kernel, bool wait, size_t size) const;

		void setCommonVariables(cl_kernel kernel, cl_uint start = 0) const;

		static void check(cl_int);

		cl_mem buffer;

	private:
		Common() { }

		static void init();
		static void finalize();

		static size_t refcount;
		static cl_context context;
		static cl_command_queue command_queue;
		static cl_program program;
		static size_t localWorkSize;
		size_t width, height, ir_width, ir_height, x, y;
	};

	template<typename Pixel, typename ClType> class Identificator : public Common
	{
	public:

		typedef Pixel Data;
		typedef ClType ClData;

		/* Create an image in the device memory from CImg image source */
		Identificator(const cimg_library::CImg<Pixel>& source) throw(APIException);

		Identificator(const size_t width, const size_t height, bool wait = true) throw(APIException) : Common(width, height)
		{
			init();
			cl_int err;
			size_t size = sizeof(Pixel) * getWidth() * getHeight();
			buffer = clCreateBuffer(getContext(), CL_MEM_READ_WRITE, size, NULL, &err);
			check(err);
		}

		/* Create an image in the device memory from the pixel vector */
		Identificator(const Pixel *source, const size_t width, const size_t height, bool wait = true) throw(APIException) : Common(width, height)
		{
			init();
			cl_int err;
			size_t size = sizeof(Pixel) * getWidth() * getHeight();
			buffer = clCreateBuffer(getContext(), CL_MEM_READ_WRITE, size, (void *)source, &err);
			check(err);

			err = clEnqueueWriteBuffer(getQueue(), buffer, wait ? CL_TRUE : CL_FALSE, 0, size, source, 0, NULL, NULL);
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
		Identificator(const Identificator<Pixel, ClType> &other) throw(APIException)
			: Common(other.getWidth(), other.getHeight()), buffer(other.buffer)
		{
			init();
			clRetainMemObject(buffer);
		}

		/* Unload the image from device memory to the buffer at dest */
		void unload(Pixel *dest, bool wait = true) const throw (APIException)
		{
			cl_int err = clEnqueueReadBuffer(getQueue(), buffer, wait ? CL_TRUE : CL_FALSE, 0,
				sizeof(Pixel) * getWidth() * getHeight(), dest, 0, NULL, NULL);
			check(err);
		}

		/* Apply threshold processing to the interest rectangle */
		void trheshold(const Pixel value, const Pixel lessValue, const Pixel moreValue, bool wait = true) throw (APIException)
		{
			setCommonVariables(thresholdKernel);

			cl_int err;
			err = clSetKernelArg(thresholdKernel, 7, sizeof(ClType), (void *)&value);
			err |= clSetKernelArg(thresholdKernel, 8, sizeof(ClType), (void *)&lessValue);
			err |= clSetKernelArg(thresholdKernel, 9, sizeof(ClType), (void *)&moreValue);
			check(err);

			execute(thresholdKernel, wait, getIRWidth() * getIRHeight());
		}

		/* Make the linear combination of interest rectangles of the current image and supplied image which
		in this case must be of the same dimensions, exception is thrown otherwise */
		void linearCombination(const Identificator<Pixel, ClType> &other, const float a, const float b, bool wait = true)
			throw (APIException)
		{
			if(other.getIRWidth() != getIRWidth() || other.getIRHeight() != getIRHeight())
				throw DimensionException(getIRWidth(), getIRHeight(), other.getIRWidth(), other.getIRHeight());

			setCommonVariables(linearCombinationKernel);
			other.setCommonVariables(linearCombinationKernel, 7);

			cl_int err;
			err = clSetKernelArg(linearCombinationKernel, 14, sizeof(cl_float), &a);
			err = clSetKernelArg(linearCombinationKernel, 15, sizeof(cl_float), &b);

			execute(linearCombinationKernel, wait, getIRWidth() * getIRHeight());
		}

		/* Make the gauss filtration of the interest rectangle and place it to dest */
		void gauss(Identificator<Pixel, ClType> &other, const float sigma, const size_t n, bool wait = true) const throw (APIException, LibraryException)
		{
			if(n > 50)
				throw LibraryException();

			if(other.getIRWidth() != getIRWidth() - 2*n || other.getIRHeight() != getIRHeight() - 2*n)
				throw DimensionException(getIRWidth() - 2*n, getIRHeight() - 2*n, other.getIRWidth(), other.getIRHeight());

			cl_int err;

			setCommonVariables(gaussKernel);
			other.setCommonVariables(gaussKernel, 7);

			err = clSetKernelArg(gaussKernel, 14, sizeof(cl_float), &sigma);
			err = clSetKernelArg(gaussKernel, 15, sizeof(cl_uint), &n);
			check(err);

			execute(gaussKernel, wait, other.getIRWidth() * other.getIRHeight());
		}

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
		Identificator operator=(const Identificator<Pixel, ClType> &rhs) { /* no, thanks */ }

		Identificator(const cl_mem otherBuffer, const Common &other) : Common(other)
		{
			init();
			cl_int err;
			size_t size = sizeof(Pixel) * getWidth() * getHeight();
			buffer = clCreateBuffer(getContext(), CL_MEM_READ_WRITE, size, NULL, &err);
			check(err);

			err = clEnqueueCopyBuffer(getQueue(), otherBuffer, buffer, 0, 0, size, 0, NULL, NULL);
			check(err);
		}

		static void init()
		{
			if(++refcount == 1)
			{
				cl_int err;
				thresholdKernel = clCreateKernel(getProgram(), (std::string("threshold") + pixelTypeSuffix).c_str(), &err);
				check(err);
				gaussKernel = clCreateKernel(getProgram(), (std::string("gauss") + pixelTypeSuffix).c_str(), &err);
				check(err);
				linearCombinationKernel = clCreateKernel(getProgram(), (std::string("linearCombination") + pixelTypeSuffix).c_str(), &err);
				check(err);
			}
		}

		static void finalize()
		{
			if(--refcount == 0)
			{
				clReleaseKernel(thresholdKernel);
				clReleaseKernel(gaussKernel);
				clReleaseKernel(linearCombinationKernel);
			}
		}

		static cl_kernel thresholdKernel, linearCombinationKernel, gaussKernel;
		static const char *pixelTypeSuffix;
		static size_t refcount;
	};

	template<typename Pixel, typename ClType> cl_kernel Identificator<Pixel, ClType>::thresholdKernel;
	template<typename Pixel, typename ClType> cl_kernel Identificator<Pixel, ClType>::linearCombinationKernel;
	template<typename Pixel, typename ClType> cl_kernel Identificator<Pixel, ClType>::gaussKernel;
	template<typename Pixel, typename ClType> size_t Identificator<Pixel, ClType>::refcount;

	typedef Identificator<byte, cl_uchar> ByteID;
	const char *ByteID::pixelTypeSuffix = "_byte";

	typedef Identificator<float, cl_float> FloatID;
	const char *FloatID::pixelTypeSuffix = "_float";

	typedef Identificator<unsigned int, cl_uint> UintID;
	const char *UintID::pixelTypeSuffix = "_uint";
}

#endif // _LIBRARY_H
