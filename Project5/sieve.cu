/*
Prime sieve code for CS 4380 / CS 5351

Copyright (c) 2017, Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is not permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <sys/time.h>

static const int ThreadsPerBlock = 256;

static __global__ void SieveInitKernel(const long top, bool array[])
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x; //todo: initialize array here
  if(idx < top)
    array[idx] = true;
}

static __global__ void SieveComputeKernel(const long top, bool array[])
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ((idx >= 2) && (idx * idx < top)) {
    long j = idx * idx;
    while (j < top){
      array[j] = false; //todo: remove multiples here
      j += idx;
    }
  }
}

static bool isPrime(long val)
{
  if (val < 2) return false;
  for (long i = 2; i * i <= val; i++) {
    if ((val % i) == 0) return false;
  }
  return true;
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  printf("Prime Sieve v1.0 [CUDA]\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "usage: %s maximum\n", argv[0]); exit(-1);}
  const long top = atol(argv[1]);
  if (top < 23) {fprintf(stderr, "error: maximum must be at least 23\n"); exit(-1);}
  printf("computing prime numbers up to but not including %ld\n", top);

  // allocate array
  bool* array = new bool[top];
  bool* array_d;
  //todo: allocate array_d here
  if(cudaSuccess != cudaMalloc((void**)&array_d, (top * sizeof(bool)))) {fprintf(stderr, "memory failed to allocate\n"); exit(-1);}

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // call kernel
  SieveInitKernel<<<(top + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(top, array_d);
  CheckCuda();
  const int sqrt_top = sqrt(top);
  SieveComputeKernel<<<(sqrt_top + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(top, array_d);
  CheckCuda();
  if(cudaSuccess != cudaMemcpy(array, array_d, top * sizeof(bool), cudaMemcpyDeviceToHost)) {fprintf(stderr, "copying from device failed\n"); exit(-1);}//todo: copy results back to CPU here

  // end time
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // print part of result
  for (long i = 2; i < 10; i++) {
    if (array[i]) {
      printf(" %ld", i);
    }
  }
  printf(" ...");
  for (long i = top - 10; i < top; i++) {
    if (array[i]) {
      printf(" %ld", i);
    }
  }
  printf("\n");

  // verify result
  if (top < 10000000) {
    for (long i = 2; i < top; i++) {
      if (array[i] != isPrime(i)) {
        fprintf(stderr, "ERROR: wrong answer for %ld\n\n", i);
        exit(-1);
      }
    }
  }

  delete [] array;
  cudaFree(array_d); //todo: free array_d here
  return 0;
}

