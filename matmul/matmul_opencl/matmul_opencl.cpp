#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <CL/cl2.hpp>

struct matsize {
    int w;
    int h;
};

const char* translateOpenCLError(cl_int errorCode)
{
    switch (errorCode)
    {
    case CL_SUCCESS:                            return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
    case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
    case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
    case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
    case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
    case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
    case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
    case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
    case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
    case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
    case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

    default:
        return "UNKNOWN ERROR CODE";
    }
}

cl_platform_id find_opencl_platform(const char* prefPlatform, cl_device_type selDeviceType) {
    std::cout << "OpenCL init is started..." << std::endl;
    std::cout << "Selected platform vendor: " << prefPlatform << std::endl;
    std::cout << "Selected device type: " << ((selDeviceType == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU") << std::endl;

    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms != 0)
    {
        std::cout << "Num of available OpenCL platforms: " << numPlatforms << std::endl;
        std::vector<cl_platform_id> platforms_ids(numPlatforms);
        clGetPlatformIDs(numPlatforms, &platforms_ids[0], NULL);

        cl_platform_id selPlatformId;
        for (cl_uint i = 0; i < numPlatforms; i++) {
            bool matchSelPlatform = true;
            cl_uint numDevices = 0;
            std::vector<char> platformName;

            if ((NULL != prefPlatform) && (strlen(prefPlatform) > 0))
            {
                size_t strLen = 0;
                clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, 0, NULL, &strLen);
                platformName.resize(strLen);
                clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, strLen, &platformName[0], NULL);

                if (strstr(&platformName[0], prefPlatform) != 0) {
                    matchSelPlatform = true;
                }
            }

            if (matchSelPlatform)
            {
                std::cout << "Using platform: " << &platformName[0] << std::endl;
                selPlatformId = platforms_ids[i];
                clGetDeviceIDs(platforms_ids[i], selDeviceType, 0, NULL, &numDevices);
                if (numDevices != 0)
                {
                    std::cout << "Num devices: " << numDevices << std::endl;
                    std::vector<cl_device_id> devices_ids(numDevices);
                    clGetDeviceIDs(platforms_ids[i], selDeviceType, numDevices, &devices_ids[0], NULL);
                    std::cout << "Devices: " << std::endl;
                    for (int j = 0; j < numDevices; j++)
                    {
                        cl::Device d(devices_ids[j]);
                        std::cout << d.getInfo<CL_DEVICE_NAME>() << std::endl;
                    }

                    return selPlatformId;
                }
                else
                {
                    std::cout << "Devices of assigned type are missing for this platform" << std::endl;
                    throw cl::Error(-32);
                }

                break;
            }
        }

        std::cout << "OpenCL init is succesful" << std::endl;
    }
    else
    {
        std::cout << "OpenCL init is failed... OpenCL platforms are missing" << std::endl;
    }

    return NULL;
}

int equalsMat(const float *m1, const float*m2,
    const matsize s1, const matsize s2) {
    if (s1.w != s2.w || s1.h != s2.h) return -1;
    int c = 0;
    for (int i = 0; i < s1.w * s1.h; i++) {
        if (m1[i] != m2[i]) c++;
    }

    return c;
}

void matmul(const float *m1, const float *m2, float *rm,
    const matsize s1, const matsize s2) {
    if (s1.w != s2.h) return;
    for (int i = 0; i < s1.h; i++) {
        for (int j = 0; j < s2.w; j++) {
            float s = 0.0;
            for (int k = 0; k < s1.w; k++) {
                s += m1[s1.w * i + k] * m2[j + s2.w * k];
            }
            rm[s2.w * i + j] = s;
        }
    }
}

void matmul_blocks(const float *m1, const float *m2, float *rm,
    const matsize s1, const matsize s2, const matsize blockSize) {
    if (s1.w != s2.h) return;

    int I, J, K;
    for (I = 0; I < s1.h; I += blockSize.h) {
        for (J = 0; J < s2.w; J += blockSize.w) {
            float *submr = rm + I * s2.w + J;
            for (K = 0; K < s1.w; K += blockSize.w) {
                const float *subm1 = m1 + I * s1.w + K;
                const float *subm2 = m2 + K * s2.w + J;

                int i, j, k;
                for (i = 0; i < blockSize.h; i++) {
                    for (j = 0; j < blockSize.w; j++) {
                        float acc = 0.0;
                        for (k = 0; k < blockSize.w; k++) {
                            acc += subm1[s1.w * i + k] * subm2[j + s2.w * k];
                        }
                        submr[s2.w * i + j] += acc;
                    }
                }
            }
        }
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");

    // init matrix
    matsize s1 = { 1024, 1024 },
        s2 = { 1024, 1024 },
        s3 = { s2.w, s1.h };
    matsize blockSize = { 4, 4 };
    float *mat1 = new float[s1.w * s1.h];
    float *mat2 = new float[s2.w * s2.h];
    float *matres1 = new float[s3.w * s3.h];
    float *matres2 = new float[s3.w * s3.h];
    float *matres3 = new float[s3.w * s3.h];

    for (int i = 0; i < s1.w * s1.h; i++) {
        mat1[i] = 2;
    }
    for (int i = 0; i < s2.w * s2.h; i++) {
        mat2[i] = 2;
    }

    memset(matres1, 0, s2.w * s1.h * sizeof(float));
    matmul_blocks(mat1, mat2, matres1, s1, s2, blockSize);

    try
    {
        // Init OpenCL
        cl_device_type selDeviceType = CL_DEVICE_TYPE_GPU;
        cl_platform_id selPlatformId = find_opencl_platform("Intel", selDeviceType);

        cl::Platform selPlatform(selPlatformId);
        std::vector<cl::Device> platformDevices;
        selPlatform.getDevices(selDeviceType, &platformDevices);
        cl::Device selDevice = platformDevices[0];
        std::cout << "Using device: " << selDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
        printf("\n");

        // Create context
        cl::Context context(selDevice);

        // Create queue
        cl::CommandQueue queue(context, selDevice);

        // Read Kernel
        std::ifstream kernel_src("matmul.cl");
        if (!kernel_src.is_open())
            throw cl::Error(1, "Cannot open file with kernel!");

        std::string str((std::istreambuf_iterator<char>(kernel_src)), std::istreambuf_iterator<char>());
        kernel_src.close();

        cl::Program::Sources sources;
        sources.push_back({ str.c_str(), str.length() });

        // Create kernel
        cl::Program program(context, sources);
        program.build({ selDevice });
        // Build with debug options
        //program.build({device}, "-g -s matmul.cl");

        // Create buffers
        cl::Buffer buffM1(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, s1.w * s1.h * sizeof(cl_float), mat1);
        cl::Buffer buffM2(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, s2.w * s2.h * sizeof(cl_float), mat2);
        cl::Buffer buffMr(context, CL_MEM_WRITE_ONLY, s3.w * s3.h * sizeof(cl_float));

        // Transfer buffers in device RAM
        //queue.enqueueWriteBuffer(buffM1, CL_TRUE, 0, s1.w * s1.h * sizeof(cl_float), mat1);
        //queue.enqueueWriteBuffer(buffM2, CL_TRUE, 0, s2.w * s2.h * sizeof(cl_float), mat2);

        int blockSize = 16;

        cl::Kernel kernel = cl::Kernel(program, "matmul");
        kernel.setArg(0, buffM1);
        kernel.setArg(1, buffM2);
        kernel.setArg(2, buffMr);
        kernel.setArg(3, s1.w);

        cl::Kernel kernel2 = cl::Kernel(program, "matmul2");
        kernel2.setArg(0, buffM1);
        kernel2.setArg(1, buffM2);
        kernel2.setArg(2, buffMr);
        kernel2.setArg(3, s1.w);
        clSetKernelArg(kernel2.get(), 4, blockSize * blockSize * sizeof(cl_float), NULL);
        clSetKernelArg(kernel2.get(), 5, blockSize * blockSize * sizeof(cl_float), NULL);

        cl::Kernel kernel3 = cl::Kernel(program, "matrixMul");
        kernel3.setArg(0, buffMr);
        kernel3.setArg(1, buffM1);
        kernel3.setArg(2, buffM2);
        clSetKernelArg(kernel3.get(), 3, blockSize * blockSize * sizeof(cl_float), NULL);
        clSetKernelArg(kernel3.get(), 4, blockSize * blockSize * sizeof(cl_float), NULL);
        kernel3.setArg(5, s1.w);
        kernel3.setArg(6, s2.w);

        cl::NDRange workSize(s3.w, s3.h);
        cl::NDRange workGroupSize(blockSize, blockSize);

        auto t = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, workSize, workGroupSize);
        queue.enqueueReadBuffer(buffMr, CL_TRUE, 0, s3.w * s3.h * sizeof(cl_float), matres3);
        printf("matmul OpenCL exec time = %lg\n",
            ((std::chrono::duration<double>) (std::chrono::high_resolution_clock::now() - t)).count());
        int c = equalsMat(matres1, matres3, s3, s3);
        printf("equals = %i\n\n", c);

        memset(matres3, 0, s2.w * s1.h * sizeof(float));
        t = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(kernel2, cl::NullRange, workSize, workGroupSize);
        queue.enqueueReadBuffer(buffMr, CL_TRUE, 0, s3.w * s3.h * sizeof(cl_float), matres3);
        printf("matmul2 OpenCL exec time = %lg\n",
            ((std::chrono::duration<double>) (std::chrono::high_resolution_clock::now() - t)).count());
        c = equalsMat(matres1, matres3, s3, s3);
        printf("equals = %i\n\n", c);

        memset(matres3, 0, s2.w * s1.h * sizeof(float));
        t = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(kernel3, cl::NullRange, workSize, workGroupSize);
        queue.enqueueReadBuffer(buffMr, CL_TRUE, 0, s3.w * s3.h * sizeof(cl_float), matres3);
        printf("matmul3 OpenCL exec time = %lg\n",
            ((std::chrono::duration<double>) (std::chrono::high_resolution_clock::now() - t)).count());
        c = equalsMat(matres1, matres3, s3, s3);
        printf("equals = %i\n", c);

        /*
        memset(matres3, 0, s2.w * s1.h * sizeof(float));
        t = omp_get_wtime();
        queue.enqueueNDRangeKernel(kernel3, cl::NullRange, workSize, workGroupSize);
        queue.enqueueReadBuffer(buffMr, CL_TRUE, 0, s3.w * s3.h * sizeof(cl_float), matres3);
        t = omp_get_wtime() - t;
        printf("matmul3 OpenCL exec time = %lg\n", t);
        c = equalsMat(matres1, matres3, s3, s3);
        printf("equals = %i\n\n", c);
        */
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << "(" << translateOpenCLError(err.err()) << ")" << std::endl;
    }
}