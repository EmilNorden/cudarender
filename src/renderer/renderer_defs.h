
#ifndef RENDERER_RENDERER_DEFS_H
#define RENDERER_RENDERER_DEFS_H

#ifdef RENDERER_CUDA_BUILD
#define DEVICE_FUNC __device__
#define HOST_FUNC __host__
#else
#define DEVICE_FUNC
#define HOST_FUNC
#endif

#endif //RENDERER_RENDERER_DEFS_H
