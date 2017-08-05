#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define NDIM 4
#define NPRIM 8
#define N1 10
#define N2 10
#define N3 10

/* Allocates a 1D array */
static void *malloc_rank1(int n1, int size)
{
	void *A;

	if ((A = (void *) malloc(n1 * size)) == NULL) {
		fprintf(stderr, "malloc failure in malloc_rank1\n");
		exit(123);
	}

	return A;
}


/* 
Allocates a 3D array of structures.

3D generalization of malloc_rank2 
*/
static void ***malloc_rank3(int n1, int n2, int n3, int size)
{
	void*** arr;
	int i,j;

	if ((arr = (void ***) malloc(n1 * sizeof(void **))) == NULL) {
		fprintf(stderr, "malloc failure in malloc_rank3\n");
		exit(124);
	}	

	for (i = 0; i < n1; i++) {
		// not sure whether the size below should be instead sizeof(void*)
		// the same goes for the next allocation
		arr[i] = (void **) malloc(n2*size);

        for(j = 0; j < n2; j++) {
        	arr[i][j] = (void *) malloc(n3 * size);
        }
    }

	return arr;
}


/* 
Allocates a multidimensional array. 
3D generalization of malloc_rank2_cont.
*/
static double ***malloc_rank3_cont(int n1, int n2, int n3)
{
	double*** arr;
	int i,j;

	arr = (double ***) malloc(n1*sizeof(double**));

	for (i = 0; i < n1; i++) {
		arr[i] = (double **) malloc(n2*sizeof(double*));

        for(j = 0; j < n2; j++) {
        	arr[i][j] = (double *) malloc(n3 * sizeof(double));
        }
    }

	return arr;
}

/* 
Initializes variables holding HARM primitives and metric.
*/
int main()
{
	int i,j,k,l,m;
	double ****p;
	struct of_geom ***geom;

	struct of_geom {
		double gcon[NDIM][NDIM];
		double gcov[NDIM][NDIM];
		double g;
	};

	/* start by allocating multidimensional arrays for each element
	of p[i]: a list of 3D arrays, each array corresponds to a
	primitive variable  */
	p = malloc_rank1(NPRIM, sizeof(double *));
	for (i = 0; i < NPRIM; i++)
		//p[i] = (double **) malloc_rank2_cont(N1, N2);
		p[i] = (double ***) malloc_rank3_cont(N1, N2, N3);
	/* then we create an array made of structures: for every array
	element there is a structure defining the metric values at
	every point */
	geom = (struct of_geom ***) malloc_rank3(N1, N2, N3, sizeof(struct of_geom));

	// initialize all variables
	// p[NPRIM][N1][N2][N3]
	for (i=0; i<NPRIM; i++) {
		for (j=0; j<N1; j++) { 
			for (k=0; k<N2; k++) { 
				for (l=0; l<N3; l++) { 
					p[i][j][k][l]=i*j*k*l;
				}
			}
		}
	}

	// geom[N1][N2][N3].gcov[4][4]
	for (i=0; i<N1; i++) {
		for (j=0; j<N2; j++) { 
			for (k=0; k<N3; k++) { 
				for (l=0; l<3; l++) { 
					for (m=0; m<3; m++) { 
						geom[i][j][k].gcov[l][m]=i*j*k*l*m;
					}
				}
			}
		}
	}	

	// prints some variables
	for (i=0; i<NPRIM; i++) {
		for (j=0; j<N1; j++) { 
			for (k=0; k<N2; k++) { 
				for (l=0; l<N3; l++) { 
					printf("%d %d %d %d %f\n", i, j, k, l, p[i][j][k][l]);
				}
			}
		}
	}

	for (i=0; i<N1; i++) {
		for (j=0; j<N2; j++) { 
			for (k=0; k<N3; k++) { 
				for (l=0; l<3; l++) { 
					for (m=0; m<3; m++) { 
						printf("%f ", geom[i][j][k].gcov[l][m]);
					}
				}
			}
		}
	}	

    return(0);
}
