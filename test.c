#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* read_file(char *filename)
{
   char* buffer = NULL;
   int string_size, read_size;
   FILE* fptr = fopen(filename, "r");

    if (fptr){
       // Set filesize seeking for last byte of the file
       fseek(fptr, 0, SEEK_END);
       string_size = ftell(fptr);
       // go back to the start of the file
       rewind(fptr);

       // Allocate memory
       buffer = (char*) malloc(sizeof(char) * (string_size + 1) );
       // Read file
       read_size = fread(buffer, sizeof(char), string_size, fptr);

       // Put a \0 in the last position so that buffer is a string
       buffer[string_size] = '\0';

       if (string_size != read_size){
           // If something goes wrong, free memory and set the buffer to NULL
           free(buffer);
           buffer = NULL;
       }

       fclose(fptr);
    }
    else{
        printf("File %s not open correctly\n", filename);
    }


    return buffer;
}
int main(int argc, char *argv[]){
    char* filename1 = "cuda.txt";
    char* filename2 = "omp.txt";

    if(argc>1){
        filename1 = argv[1];
        filename2 = argv[2];
    }

    char* file1 = read_file(filename1);
    char* file2 = read_file(filename2);

    if (file1 && file2)
    {
        char* equal = strcmp(file1, file2) == 0? "" : "not ";
        printf("Files %s and %s are %sequal\n", filename1, filename2, equal);
    }
    else{
        printf("Files not read correctly\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
