#include <stdio.h>



void main(){
	int i = 0;
	int stop = 5;
	while(i < stop){
		here:
		printf("i = %d\n", i);
		i++;
	}
	stop = 10;
	if(i < 10){
		printf("goto is working!\n");
		goto here;
	}
}