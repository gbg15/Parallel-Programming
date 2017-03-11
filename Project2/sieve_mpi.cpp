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
#include <sys/time.h>
#include <mpi.h>

static bool isPrime(long val)
{
  if (val < 2) return false;
  for (long i = 2; i * i <= val; i++) {
    if ((val % i) == 0) return false;
  }
  return true;
}

int main(int argc, char *argv[])
{
  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank == 0)	
  	printf("Prime Sieve v1.0 [MPI]\n");



  // check command line
  if (argc != 2) {fprintf(stderr, "usage: %s maximum\n", argv[0]); exit(-1);}
  const long top = atol(argv[1]);
  if (top < 23) {fprintf(stderr, "error: maximum must be at least 23\n"); exit(-1);}
  
  if(my_rank == 0) 
    printf("computing prime numbers up to but not including %ld\n", top);

  // allocate array
  bool* array = new bool[top];

  // array for process 0
  bool* array2 = NULL;
  if(my_rank == 0)
  	array2 = new bool[top];

  MPI_Barrier(MPI_COMM_WORLD);

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // initialize array
  for (long i = 2; i < top; i++) {
    array[i] = true;
    if(my_rank == 0)
      array2[i] = true;
  }

  // parallelized remove multiples
  for(long i = 2 + my_rank; (i*i) < top; i += comm_sz){
  	long j = i * i;
  	while(j < top){
  		array[j] = false;
  		j += i;
  	}
  }


  MPI_Reduce(array, array2, top, MPI::BOOL, MPI_LAND, 0, MPI_COMM_WORLD);

  // end time
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  if(my_rank == 0)
    printf("compute time: %.4f s\n", runtime);

  // print part of result
  if(my_rank == 0){
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
  }  

  // verify result
  if(my_rank == 0){
    if (top < 10000000) {
      for (long i = 2; i < top; i++) {
        if (array2[i] != isPrime(i)) {
          fprintf(stderr, "ERROR: wrong answer for %ld\n\n", i);
          exit(-1);
        }
      }
    }
  }

  delete [] array;
  if(my_rank == 0)
  	delete [] array2;
  MPI_Finalize();
  return 0;
}
