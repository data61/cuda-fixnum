#pragma once

#include "cuda_wrap.h"

// TODO: Check whether the synchronize calls are necessary here (they
// are clearly sufficient). Maybe they should be cudaStreamSynchronize()?
struct managed {
    void *operator new(size_t bytes) {
        void *ptr;
        cuda_malloc_managed(&ptr, bytes);
        cuda_device_synchronize();
        return ptr;
    }

    void operator delete(void *ptr) {
        cuda_device_synchronize();
        cuda_free(ptr);
    }
};
