/*
Fractal code for CS 4380 / CS 5351

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
#include <sys/time.h>
#include "cs43805351.h"
#include <mpi.h>
#include <math.h>

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;

int main(int argc, char *argv[])
{
  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank==0)
    printf("Fractal v1.6 [MPI]\n");

  // check command line
  
    if (argc != 3) {fprintf(stderr, "usage: %s frame_width num_frames\n", argv[0]); exit(-1);}
    int width = atoi(argv[1]);
    if (width < 10) {fprintf(stderr, "error: frame_width must be at least 10\n"); exit(-1);}
    int frames = atoi(argv[2]);
    if (frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
    if(my_rank == 0) {printf("computing %d frames of %d by %d fractal using %d cores\n", frames, width, width, comm_sz);}
   
    // print error message
    if(frames % comm_sz != 0){
      printf("Frames are not a multiple of the number of processes, terminating program\n");
      exit(-1);  
    }
  

  // print processes
  if(my_rank == 0)
    printf("The total number of processes is: %d\n", comm_sz);

  // start and end
  int my_start, my_end, mframes;
  int framePP =  (long)frames / comm_sz; // frames per process
  my_start = my_rank * framePP;
  my_end = ((my_rank + 1) * framePP - 1);
  mframes= my_end - my_start +1;
  

  // allocate picture array
  unsigned char* pic = new unsigned char[framePP * width * width];
  // allocate array for process 0
  unsigned char* masterPic = NULL;
  if(my_rank == 0)
    masterPic = new unsigned char[frames * width * width];

  MPI_Barrier(MPI_COMM_WORLD); //Barrier before time starts
  // start time
  timeval start, end;
  if(my_rank == 0) {gettimeofday(&start, NULL);}

  // compute frames
  double delta = Delta * pow(0.98, my_start);
  for (int frame = 0; frame < mframes; frame++) {
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    for (int row = 0; row < width; row++) {
      const double cy = yMin + row * dw;
      for (int col = 0; col < width; col++) {
        const double cx = xMin + col * dw;
        double x = cx;
        double y = cy;
        int depth = 256;
        double x2, y2;
        do {
          x2 = x * x;
          y2 = y * y;
          y = 2 * x * y + cy;
          x = x2 - y2 + cx;
          depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        pic[frame * width * width + row * width + col] = (unsigned char)depth;
      }
    }
    delta *= 0.98;
  }

  // gather frames to process 0
  MPI_Gather(pic, framePP * width * width, MPI_UNSIGNED_CHAR, masterPic, framePP * width * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  
  // end time
  if(my_rank == 0){
    gettimeofday(&end, NULL);
    double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
    printf("compute time: %.4f s\n", runtime);

    // verify result by writing frames to BMP files
    if ((width <= 256) && (frames <= 64)) {
      for (int frame = 0; frame < frames; frame++) {
        char name[32];
        sprintf(name, "fractal%d.bmp", frame + 1000);
        writeBMP(width, width, &masterPic[frame * width * width], name);
      }
    }
  }

  delete [] pic;
  if(my_rank == 0)
    delete [] masterPic;
  MPI_Finalize();
  return 0;
}
