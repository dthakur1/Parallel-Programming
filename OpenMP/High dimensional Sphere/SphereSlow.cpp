#include <stdio.h>
#include <cmath>
#include <iostream>
#include <random>
#include <omp.h>
#include <cstdlib>
#include <ctime>


void putPointInSlot(float Slots[], float distance){
	int SlotNumber = int(distance * 100);
	Slots[SlotNumber] = Slots[SlotNumber]+1;
}



int main(int argc, char* argv[]){

if(argc < 2){
	printf("Usage: ./Sphere numThreads\n");
	exit(0);
}
int numThreads = atoi(argv[1]);

printf("numThreads : %d\n", numThreads);
omp_set_num_threads(numThreads);


int totalPoints = 1000;

//FILE *out = fopen("OutputFast.txt", "w+");

for(int dim=2; dim<=16; dim++)
{

int Dimension = dim;
float Slots[101];
for(int slt=0; slt<101; slt++)
	Slots[slt] = 0.0;

float distances[totalPoints];
int partitionSize = (totalPoints / numThreads);


#pragma omp parallel
{
std::uniform_real_distribution<float> dist(-1, 1);
std::default_random_engine eng(omp_get_thread_num() + 13);
int tdnum = omp_get_thread_num();
float localDistances[partitionSize];

for(int i= 0; i<partitionSize; i++)
{					
	float sum, cord;
	int gotPoint = 0;
	while(gotPoint != 1){
		sum = 0.0;
		for(int j=0; j<Dimension; j++){
			cord = dist(eng);
			sum += cord * cord;
			}
		if( (cord = sqrt(sum))  <= 1.0){
			localDistances[i] = cord;
			gotPoint = 1;			
		}
	}

}

for(int i= 0; i< partitionSize; i++){
	distances[(tdnum*partitionSize) + i] = localDistances[i];
}

}


for(int i=0; i<totalPoints-1; i++){
	putPointInSlot(Slots, distances[i]);		
}

Slots[99] = Slots[99] + Slots[100];		//points ON the unit sphere

for(int i=0; i<100; i++){
	Slots[i] = float(Slots[i])/ float(totalPoints) * 100 ;
//	fprintf(out, "%d,%f,%d,", i,Slots[i],Dimension);
	printf("Dimension :%d Slot %d : %f  \n",Dimension, i, Slots[i]);
}
//fflush(out);
}
return 0;

}


