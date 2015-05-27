#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <assert.h>

int num_cities;
int **distances;
int *visited;
int cost = 0;


void get_input(char filename[], char n[]){
    FILE * fp;
    int i ,j;  //for the for loops
    
    //attempt to open file...
    fp = fopen(filename, "r");

    //get number of cities
    sscanf(n, "%d", &num_cities);

    
    //if file cannot be opened, do the following...
    if(!fp){
        printf("Cannot open file %s\n", filename);
        exit(1);
    }
    
    
    /* Now, time to allocate the matrices and vectors */
    distances = (int**)malloc(num_cities * sizeof(int*));

    if( !distances){
        printf("Cannot allocate a!\n");
        exit(1);
    }

    for(i = 0; i < num_cities; i++){
        distances[i] = (int *)malloc(num_cities * sizeof(int)); 
        if( !distances[i]){
            printf("Cannot allocate distances[%d]!\n",i);
            exit(1);
        }
    }

    
    ///////////////////////////////////////////////////////////
    /* Now .. Filling the blanks */ //////////////////////////

    for(i = 0; i < num_cities; i++){
        for(j = 0; j < num_cities; j++){
            fscanf(fp,"%d ",&distances[i][j]);
        }
    }
    
    /*for(i = 0; i < num_cities; i++){
    	printf("\n");
        for(j = 0; j < num_cities; j++){
            printf("%d ",distances[i][j]);      
    	}
	}*/
	
	visited = malloc(num_cities * sizeof(int));
	for(i = 0; i < num_cities; i++) {
		visited[i] = 0;
		//printf("%d ", visited[i]);
	}
    //close the file....
    fclose(fp);
}

void mincost(int city) {
	int ncity;
	visited[city] = 1;
	printf("%d ",city);
	ncity = least(city);
	if(ncity == 10000)
	{
		//ncity=0;
		//printf("%d",ncity);
		//cost += distances[city][ncity];
		return;
	}
	mincost(ncity);
}

int least(int c) {
	int i,nc=10000;
	int min=10000,kmin;
	#pragma omp parallel for
	for(i=0;i<num_cities;i++)
	{
		if((distances[c][i]!=0)&&(visited[i]==0))
			if(distances[c][i]<min) {
				min = distances[c][i];
				kmin = distances[c][i];
				nc = i;
				}
	}
	if(min!=10000)
	//#pragma omp critical
	cost+=kmin;
	return nc;
}

void printCost() {
	printf("\nMinimum cost: ");
	printf("%d\n",cost);
}


int main(int argc, char *argv[]){   
    int i; 
    //int thread_count = strtol(argv[1], NULL, 10);

    //check to see if there are 3 args at the command line...
    if( argc != 3) {
        printf("Usage: newTSM: filename, number of cities\n");
        exit(1);
    }
    
    /* Read the input file and fill the global data structure above */
    //argv[1] = file, argv[2] = num of cities 
    get_input(argv[1], argv[2]);
    
    printf("\nThe Path is:\n");
	mincost(0);
	printCost();
    
    //printf("Does this work?\n");
    
    //#pragma omp num_thread(thread_count)

    exit(0);
}
