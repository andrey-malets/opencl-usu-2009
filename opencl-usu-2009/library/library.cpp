#include "library.h"
#include "CL/cl.h"

namespace opencl_usu_2009
{
	Common::Common(size_t width, size_t height)
	{
		this->width = width;
		this->height = height;
		setInterestRect(0, 0, width, height);

		if(++refcount == 1)
			init();
	}

	Common::Common(const Common &other)
		: width(other.width), height(other.height),
		x(other.x), y(other.y),
		ir_width(other.ir_width), ir_height(other.ir_height)
	{
		++refcount;
	}

	Common::~Common()
	{
		if(--refcount == 0)
			finalize();
	}

	void Common::setInterestRect(const size_t x0, const size_t y0, const size_t width, const size_t height) throw (DimensionException)
	{
		if(x0+width > this->width || y0+height > this->height)
			throw DimensionException();
		x = x0;
		y = y0;
		ir_width = width;
		ir_height = height;
	}

	void Common::init()
	{
		cl_int err;
		context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
		check(err);

		size_t device_count;
		err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &device_count);
		check(err);

		cl_device_id *devs = new cl_device_id[device_count];
		err = clGetContextInfo(context, CL_CONTEXT_DEVICES, device_count, devs, NULL);
		try { check(err); }
		catch(...)
		{
			delete[] devs;
			throw;
		}

		command_queue = clCreateCommandQueue(context, devs[0], 0, &err);
		try { check(err); }
		catch(...)
		{
			delete[] devs;
			throw;
		}

		std::ifstream in(kernelsFile);
		if(!in)
			throw LibraryException();

		std::string buf;
		std::string buffer;
		while(std::getline(in, buf))
		{
			buffer.append(buf);
			buffer.append("\n");
		}
		
		const char *ptr = buffer.c_str();
		program = clCreateProgramWithSource(getContext(), 1, &ptr, NULL, &err);
		check(err);

		err = clBuildProgram(program, 1, devs, "", NULL, NULL);
		check(err);
		delete[] devs;

		thresholdKernel = clCreateKernel(program, (std::string("threshold") + pixelTypeSuffix).c_str(), &err);
		check(err);

		gaussKernel = clCreateKernel(program, (std::string("gauss") + pixelTypeSuffix).c_str(), &err);
		check(err);

		linearCombinationKernel = clCreateKernel(program, (std::string("linearCombination") + pixelTypeSuffix).c_str(), &err);
		check(err);
	}

	void Common::finalize()
	{
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		clReleaseProgram(program);
	}

	void Common::check(cl_int code)
	{
		/* TODO: init appropriate error code */
		if(code != CL_SUCCESS) throw APIException();
	}

	cl_command_queue Common::command_queue;
	cl_context Common::context;
	cl_program Common::program;
	size_t Common::refcount = 0;
	const char *Common::kernelsFile = "library.cl";
	cl_kernel Common::thresholdKernel;
	cl_kernel Common::linearCombinationKernel;
	cl_kernel Common::gaussKernel;

	const char *Identificator<byte>::pixelTypeSuffix = "_byte";

	void Identificator<byte>::trheshold(const byte value, const byte lessValue, const byte moreValue)
	{
		cl_int err;
		err = clSetKernelArg(thresholdKernel, 0, sizeof(cl_mem), (void *)&buffer);
		err |= clSetKernelArg(thresholdKernel, 1, sizeof(cl_int), (void *)(getWidth()*getHeight()));
		err |= clSetKernelArg(thresholdKernel, 2, sizeof(cl_uchar), (void *)&value);
		err |= clSetKernelArg(thresholdKernel, 3, sizeof(cl_uchar), (void *)&lessValue);
		err |= clSetKernelArg(thresholdKernel, 4, sizeof(cl_uchar), (void *)&moreValue);
		check(err);

		err = clEnqueueTask(getQueue(), thresholdKernel, 0, NULL, NULL);
		check(err);
	}
}
