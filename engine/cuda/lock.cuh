#ifndef LOCK_CUH_
#define LOCK_CUH_

struct Lock {
	int *mutex;

	Lock();
	Lock(const Lock &);

	~Lock();

	__device__ void lock();
	__device__ void unlock();
};

Lock::Lock()
{
	int tmp = 0;

	cudaMalloc(&mutex, sizeof(int));
	cudaMemcpy(mutex, &tmp, sizeof(int),
			cudaMemcpyHostToDevice);
}

Lock::Lock(const Lock &other)
{
	cudaMalloc(&mutex, sizeof(int));
	cudaMemcpy(mutex, other.mutex, sizeof(int),
			cudaMemcpyDeviceToDevice);
}

Lock::~Lock()
{
	cuda_device_free(mutex);
}

__device__
void Lock::lock()
{
	while (atomicCAS(mutex, 0, 1) != 0);
}

__device__
void Lock::unlock()
{
	atomicExch(mutex, 0);
}

#endif
