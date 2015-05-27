#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_RAND 100001

int size;

int setup(char n[]){
	int i;
	int max = 0;
	int *rand_num;
	time_t t;
	
	//get size of array
	sscanf(n, "%d", &size);
	
	
	srand((unsigned) time(&t));
	
	
	rand_num = malloc(size * sizeof(int));

    
	for(i = 0; i < size; i++){
		rand_num[i] = random() % MAX_RAND;
		//printf("%d\n", i);
	}
	
	for(i = 0; i < size; i++) {
		if(rand_num[i] > max) {
			max = rand_num[i];
		}
	}

return max;	
	
}

int main(int argc, char *argv[]){ 

	int i; 
	
	int result = setup(argv[1]);
	
	printf("The max value in the array is: %d\n", result);
	
	exit(0);
}
	
	
