/*
N-body code for CS 4380 / CS 5351

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
#include <cmath>
#include <sys/time.h>
#include <cuda.h>
#include "cs43805351.h"

static const int ThreadsPerBlock = 256;
static const int WIDTH = 512;

struct Data {
  float mass, posx, posy, posz, velx, vely, velz, accx, accy, accz;
};

static void output(const int nbodies, const Data* const data, const int step)
{
  unsigned char* bmp = new unsigned char[WIDTH * WIDTH];
  for (int i = 0; i < WIDTH * WIDTH; i++) {
    bmp[i] = 0;
  }

  for (int i = 0; i < nbodies; i++) {
    float fz = data[i].posz + 3.0f;
    if (fz > 0) {
      float fx = data[i].posx;
      float fy = data[i].posy;
      float dsqr = fx * fx + fy * fy + fz * fz;
      int x = atanf(fx / fz) * (WIDTH / 2) + (0.5f + WIDTH / 2);
      int y = atanf(fy / fz) * (WIDTH / 2) + (0.5f + WIDTH / 2);
      int c = 255.5f - dsqr * 5.0f;
      if (c > 255) c = 255;
      if (c < 64) c = 64;
      if ((0 <= x) && (x < WIDTH) && (0 <= y) && (y < WIDTH)) {
        if (c > bmp[x + y * WIDTH]) bmp[x + y * WIDTH] = c;
      }
    }
  }

  char name[32];
  sprintf(name, "nbody%d.bmp", step + 1000);
  writeBMP(WIDTH, WIDTH, bmp, name);
  delete [] bmp;
}

/******************************************************************************/
/*** generate input (based on SPLASH2) ****************************************/
/******************************************************************************/

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;

static void drndset(const int seed)
{
  A = 1;
  B = 0;
  randx = (A * seed + B) & MASK;
  A = (MULT * A) & MASK;
  B = (MULT * B + ADD) & MASK;
}

static double drnd()
{
  lastrand = randx;
  randx = (A * randx + B) & MASK;
  return lastrand / TWOTO31;
}

static void generateInput(const int nbodies, Data* const data)
{
  drndset(7);
  double rsc = (3 * 3.1415926535897932384626433832795) / 16;
  double vsc = sqrt(1.0 / rsc);
  for (int i = 0; i < nbodies; i++) {
    data[i].mass = 1.0 / nbodies;
    double r = 1.0 / sqrt(pow(drnd() * 0.999, -2.0 / 3.0) - 1);
    double x, y, z, sq;
    do {
      x = drnd() * 2.0 - 1.0;
      y = drnd() * 2.0 - 1.0;
      z = drnd() * 2.0 - 1.0;
      sq = x * x + y * y + z * z;
    } while (sq > 1.0);
    double scale = rsc * r / sqrt(sq);
    data[i].posx = x * scale;
    data[i].posy = y * scale;
    data[i].posz = z * scale;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x * x * pow(1 - x * x, 3.5));
    double v = x * sqrt(2.0 / sqrt(1 + r * r));
    do {
      x = drnd() * 2.0 - 1.0;
      y = drnd() * 2.0 - 1.0;
      z = drnd() * 2.0 - 1.0;
      sq = x * x + y * y + z * z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    data[i].velx = x * scale;
    data[i].vely = y * scale;
    data[i].velz = z * scale;
  }
  for (int i = 0; i < nbodies; i++) {
    data[i].accx = 0.0f;
    data[i].accy = 0.0f;
    data[i].accz = 0.0f;
  }
}

/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

static __global__ void forceCalculation(const int nbodies, Data* const data, const int step, const float dthf, const float epssq)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < nbodies) {
    const float px = data[i].posx;//todo: compute force of one body here
    const float py = data[i].posy;
    const float pz = data[i].posz;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int j = 0; j < nbodies; j++) {
      const float dx = data[j].posx - px;
      const float dy = data[j].posy - py;
      const float dz = data[j].posz - pz;
      float tmp = dx * dx + dy * dy + dz * dz;
      tmp = 1.0f / sqrtf(tmp + epssq);
      tmp = data[j].mass * tmp * tmp * tmp;
      ax += dx * tmp;
      ay += dy * tmp;
      az += dz * tmp;
    }

    if (step > 0) {
      data[i].velx += (ax - data[i].accx) * dthf;
      data[i].vely += (ay - data[i].accy) * dthf;
      data[i].velz += (az - data[i].accz) * dthf;
    }

    data[i].accx = ax;
    data[i].accy = ay;
    data[i].accz = az;
  }
}

/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/

static __global__ void integration(const int nbodies, Data* const data, const float dthf)
{
  //todo: perform integration here
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const float dtime = dthf * 2.0f;
  if(i<nbodies){
    const float dvelx = data[i].accx * dthf;
    const float dvely = data[i].accy * dthf;
    const float dvelz = data[i].accz * dthf;

    const float velhx = data[i].velx + dvelx;
    const float velhy = data[i].vely + dvely;
    const float velhz = data[i].velz + dvelz;

    data[i].posx += velhx * dtime;
    data[i].posy += velhy * dtime;
    data[i].posz += velhz * dtime;

    data[i].velx = velhx + dvelx;
    data[i].vely = velhy + dvely;
    data[i].velz = velhz + dvelz;
  }
}

/******************************************************************************/

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
  printf("N-body v1.0 [CUDA]\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "usage: %s number_of_bodies number_of_timesteps\n", argv[0]); exit(-1);}
  const int nbodies = atoi(argv[1]);
  if (nbodies < 10) {fprintf(stderr, "error: number_of_bodies must be at least 10\n"); exit(-1);}
  const int timesteps = atoi(argv[2]);
  if (timesteps < 1) {fprintf(stderr, "error: number_of_timesteps must be at least 1\n"); exit(-1);}
  printf("simulating %d bodies for %d time steps\n", nbodies, timesteps);

  // allocate and initialize data
  const float dthf = 0.025f * 0.5f;
  const float epssq = 0.05f * 0.05f;
  Data* data = new Data[nbodies];
  generateInput(nbodies, data);
  Data* data_d;
  if (cudaSuccess != cudaMalloc((void **)&data_d, nbodies * sizeof(Data))) {fprintf(stderr, "could not allocate memory\n"); exit(-1);}

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // compute result for each time step
  if (cudaSuccess != cudaMemcpy(data_d, data, nbodies * sizeof(Data), cudaMemcpyHostToDevice)) {fprintf(stderr, "copying to device failed\n"); exit(-1);}
  for (int step = 0; step < timesteps; step++) {
    int blockSize = (nbodies * ThreadsPerBlock - 1) / ThreadsPerBlock; //todo: launch forceCalculation kernel here
    forceCalculation<<<blockSize, ThreadsPerBlock>>>(nbodies, data_d, step, dthf, epssq);
    CheckCuda();
    integration<<<blockSize, ThreadsPerBlock>>>(nbodies, data_d, dthf); //todo: launch integration kernel here
    CheckCuda();
    //if (cudaSuccess != cudaMemcpy(data, data_d, nbodies * sizeof(Data), cudaMemcpyDeviceToHost)) {fprintf(stderr, "copying from device failed\n"); exit(-1);}
    //output(nbodies, data, step);
  }
  if (cudaSuccess != cudaMemcpy(data, data_d, nbodies * sizeof(Data), cudaMemcpyDeviceToHost)) {fprintf(stderr, "copying from device failed\n"); exit(-1);}

  // end time
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  delete [] data;
  cudaFree(data_d);
  return 0;
}

