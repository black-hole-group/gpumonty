
extern "C" __device__
int gpu_thread_id() {
    return (blockDim.x * blockIdx.x) + threadIdx.x;
}
