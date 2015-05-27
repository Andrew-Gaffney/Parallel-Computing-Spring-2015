#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */
int comm_sz;
int my_rank;
MPI_Comm comm = MPI_COMM_WORLD;



/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*));
 if( !a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }

 for(i = 0; i < num; i++) 
  {
    a[i] = (float *)malloc(num * sizeof(float)); 
    if( !a[i])
  	{
		printf("Cannot allocate a[%d]!\n",i);
		exit(1);
  	}
  }
 
 x = (float *) malloc(num * sizeof(float));
 if( !x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }

 /*curr = (float *) malloc(num * sizeof(float));
 if( !curr)
  {
	printf("Cannot allocate curr!\n");
	exit(1);
  }*/

 b = (float *) malloc(num * sizeof(float));
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Now .. Filling the blanks */ 

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
 
 fclose(fp); 

}

/* Herein lies my attempt at parallelizing the code, it failed. It also results
in a seg fault because I couldn't figure out how to properly split and 
manipulate the data in parallel.

/************************************************************/

int main(int argc, char *argv[])
{

 int i;
 int nit = 0; /* number of iterations */
 
 
 MPI_Init(NULL, NULL);
 
  /* Get the number of processes */
 MPI_Comm_size(comm, &comm_sz); 

   /* Get my rank among all the processes */
 MPI_Comm_rank(comm, &my_rank); 

 
 if(argc != 2)
 {
   printf("Usage: gsref filename\n");
   exit(1);
 }
 
 
 /* Read the input file and fill the global data structure above */ 
 
 get_input(argv[1]);
 
 /* Check for convergence condition */
 check_matrix();
	
/* Herein lies my attempt at parallelizing the code, it failed. It also results
in a seg fault with more than 1 process because I couldn't figure out how to properly split and manipulate the data in parallel.*/
	
	int j = 0;
	float total, c, divisor, xNew;
	float newErr = 1;
	int sendCt;
	
	float** rec_coef = NULL; 
	float* rec_val = NULL;
	float* rec_con = NULL;
	
	
	rec_coef = (float **)malloc(num * num * num * sizeof(float));
	rec_val = (float *)malloc(num * sizeof(float));
	rec_con = (float *)malloc(num * sizeof(float));
	
	if(num%comm_sz != 0)
		sendCt = (num/comm_sz) + 1;
	else if(num < comm_sz)
		sendCt = 1;
	else
		sendCt = num/comm_sz;
	
	MPI_Scatter(a, num*sendCt, MPI_FLOAT, rec_coef, num*sendCt, MPI_FLOAT, 0, comm);
	MPI_Scatter(x, sendCt, MPI_FLOAT, rec_val, sendCt, MPI_FLOAT, 0, comm);	
	MPI_Scatter(b, sendCt, MPI_FLOAT, rec_con, sendCt, MPI_FLOAT, 0, comm);
			
	while(newErr > err){
		nit++;
		for(i = 0; i < sendCt; i++) {
			c = rec_con[i];
			divisor = rec_coef[i][i];
			total = 0;
			for(j = 0; j < num; j++) {
				if(i != j) {
					total += rec_val[j] * rec_coef[i][j];
				}
			}
			xNew = (c - total)/divisor;
			newErr = fabs((xNew - rec_val[i])/xNew);
			rec_val[i] = xNew;
			//printf("%f\n", rec_val[i]);
		}
	}
	
	//printf("%d\n", i);
	
	MPI_Gather(rec_val, sendCt, MPI_FLOAT, x, sendCt, MPI_FLOAT, 0, comm);
	
	free(rec_coef);
	free(rec_con);
	free(rec_val);

	
 /* Writing to the stdout */
 /* Keep that same format */
 
 if(my_rank == 0) {
 for( i = 0; i < num; i++)
   printf("%f\n",x[i]);
 
 printf("total number of iterations: %d\n", nit);
 }
 
 MPI_Finalize();
 
 exit(0);
}
