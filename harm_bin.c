/* 
Reads the header line as a string, and then read the 42 floats 
from a HARMPI binary file, prints them to the screen.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    FILE *binfile;
    int i;
    const int n=42;
    float x[n];
    char header[1024], fname[1024];

    // handle command-line argument
    if ( argc != 2 ) {
        printf( "usage: %s filename \n", argv[0] );
        exit(0);
    } 
    
    // reads command-line argument 
    strncpy(fname, argv[1], 1024); 
 
    // opens binary file
    binfile = fopen(fname, "rb");

    // reads header line
    fgets(header, 1024, binfile);
    // prints header 
    printf("Header: %s\n", header);

    // reads 42 floats from binary data
    fread(x, sizeof(float), n, binfile);

    // content of first element of arrays
    printf("\nBinary data:\n");
    for (i = 0; i < n; i++) {
        printf("%e ", x[i]);
    }
    printf("\n");

    fclose(binfile);

    return 0;
}