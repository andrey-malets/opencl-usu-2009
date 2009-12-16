#include "library.h"
#include <vector>

namespace opencl_usu_2009
{
	class DeviceVector
	{
	public:
		DeviceVector(size_t capacity) : capacity(capacity)
		{
			if(capacity == 0)
				throw LibraryException();
			ptr = new cl_device_id[capacity];
		}

		~DeviceVector() { delete[] ptr; }
		cl_device_id *p() { return ptr; }
		cl_device_id get() { return ptr[0]; }
	private:
		cl_device_id *ptr;
		size_t capacity;
	};

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
			throw DimensionException(this->width, this->height, x0+width, y0+height);

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

		DeviceVector devs(device_count);
		err = clGetContextInfo(context, CL_CONTEXT_DEVICES, device_count, devs.p(), NULL);
		check(err);

		command_queue = clCreateCommandQueue(context, devs.get(), 0, &err);
		check(err);

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

		err = clBuildProgram(program, 1, devs.p(), buildOptions, NULL, NULL);
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
		if(code != CL_SUCCESS) throw APIException(code);
	}

	void Common::setCommonVariables(cl_kernel kernel, cl_uint start) const
	{
		cl_int err;
		err = clSetKernelArg(kernel, start, sizeof(cl_mem), (void *)&buffer);
		err |= clSetKernelArg(kernel, start + 1, sizeof(cl_uint), (void *)&width);
		err |= clSetKernelArg(kernel, start + 2, sizeof(cl_uint), (void *)&height);
		err |= clSetKernelArg(kernel, start + 3, sizeof(cl_uint), (void *)&x);
		err |= clSetKernelArg(kernel, start + 4, sizeof(cl_uint), (void *)&y);
		err |= clSetKernelArg(kernel, start + 5, sizeof(cl_uint), (void *)&ir_width);
		err |= clSetKernelArg(kernel, start + 6, sizeof(cl_uint), (void *)&ir_height);
		check(err);
	}

	void Common::execute(cl_kernel kernel, bool wait, size_t size) const
	{
		size_t global = (((localWorkSize-1+size)/localWorkSize)*localWorkSize);
		cl_int err = clEnqueueNDRangeKernel(getQueue(), kernel, 1, NULL, &global, &localWorkSize, 0, NULL, NULL);
		check(err);

		if(wait)
		{
			err = clFinish(getQueue());
			check(err);
		}
	}

	cl_command_queue Common::command_queue;
	cl_context Common::context;
	cl_program Common::program;
	size_t Common::refcount = 0;
	size_t Common::localWorkSize = 256;
	const char *Common::kernelsFile = "library.cl";
	const char *Common::buildOptions = "";
}
