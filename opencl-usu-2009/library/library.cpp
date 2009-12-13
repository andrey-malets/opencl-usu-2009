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
		delete[] devs;
	}

	void Common::finalize()
	{
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
	}

	void Common::check(cl_int code)
	{
		/* TODO: init appropriate error code */
		if(code != CL_SUCCESS) throw APIException();
	}

	cl_command_queue Common::command_queue;
	cl_context Common::context;
	size_t Common::refcount = 0;
}
