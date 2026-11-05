// OpenCL devices of this machine with their extensions and CL_DEVICE_DOUBLE_FP_CONFIG (notes/GPU_FEASIBILITY.md)
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <stdio.h>
int main(void) {
    cl_platform_id p; cl_uint np = 0;
    if (clGetPlatformIDs(1, &p, &np) != CL_SUCCESS || np == 0) { printf("no OpenCL platform\n"); return 1; }
    char buf[8192];
    clGetPlatformInfo(p, CL_PLATFORM_VERSION, sizeof buf, buf, NULL); printf("platform: %s\n", buf);
    cl_device_id d[8]; cl_uint nd = 0;
    clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 8, d, &nd);
    for (cl_uint i = 0; i < nd; i++) {
        cl_device_fp_config fp = 0;
        clGetDeviceInfo(d[i], CL_DEVICE_NAME, sizeof buf, buf, NULL); printf("device %u: %s\n", i, buf);
        clGetDeviceInfo(d[i], CL_DEVICE_VERSION, sizeof buf, buf, NULL); printf("  version: %s\n", buf);
        clGetDeviceInfo(d[i], CL_DEVICE_EXTENSIONS, sizeof buf, buf, NULL); printf("  extensions: %s\n", buf);
        clGetDeviceInfo(d[i], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof fp, &fp, NULL); printf("  double fp config: 0x%llx\n", (unsigned long long)fp);
    }
    return 0;
}
