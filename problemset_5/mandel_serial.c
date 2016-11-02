#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

/* Shorthand for less typing */
typedef unsigned char uchar;

/* Declarations of output functions */
void output();
void fancycolour(uchar *p, int iter);
void savebmp(char *name, uchar *buffer, int x, int y);

/* Struct for complex numbers */
typedef struct {
  double real, imag;
} complex_t;

/* Size of image, in pixels */
const int XSIZE = 2560;
const int YSIZE = 2048;

/* Max number of iterations */
const int MAXITER = 255;

/* Range in x direction */
double xleft = -2.0;
double xright = 1.0;
double ycenter = 0.0;

/* Range in y direction, calculated in main
 * based on range in x direction and image size
 */
double yupper, ylower;

/* Distance between numbers */
double step;

/* Global array for iteration counts/pixels */
int* pixel;


/* Only for serial timings */
double walltime() {
    static struct timeval t;
    gettimeofday(&t, NULL);
    return (t.tv_sec + 1e-6 * t.tv_usec);
}

/* Calculate the number of iterations until divergence for each pixel.
 * If divergence never happens, return MAXITER
 */
void calculate() {
  for (int i = 0; i < XSIZE; i++) {
    for (int j = 0; j < YSIZE; j++) {
      complex_t c, z, temp;
      int iter = 0;
      c.real = (xleft + step * i);
      c.imag = (ylower + step * j);
      z = c;
      while (z.real * z.real + z.imag * z.imag < 4) {
        temp.real = z.real * z.real - z.imag * z.imag + c.real;
        temp.imag = 2 * z.real * z.imag + c.imag;
        z = temp;
        iter++;
        if(iter == MAXITER){
            break;
        }
      }
      pixel[j * XSIZE + i] = iter;
    }
  }
}


int main(int argc, char **argv) {
    
    /* Check input arguments */
  if (argc == 1) {
    puts("Usage: MANDEL n");
    puts("n decides whether image should be written to disk (1 = yes, 0 = no)");
    return 0;
  }
  
  /* Calculate the range in the y - axis such that we preserve the aspect ratio */
  step = (xright - xleft)/XSIZE;
  yupper = ycenter + (step * YSIZE)/2;
  ylower = ycenter - (step * YSIZE)/2;
  
  /* Allocate memory for the entire image */
  pixel = (int*) malloc(sizeof(int) * XSIZE * YSIZE);
  

  /* Perform calculation */
  calculate();

  /* Output */
  if (strtol(argv[1], NULL, 10) != 0) {
      output();
  }
  
  return 0;
}

/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down */
void savebmp(char *name, uchar *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
  if (!f) {
    printf("Error writing image to disk.\n");
    return;
  }
  unsigned int size = x * y * 3 + 54;
  uchar header[54] = {'B', 'M',
                      size&255,
                      (size >> 8)&255,
                      (size >> 16)&255,
                      size >> 24,
                      0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x&255, x >> 8, 0,
                      0, y&255, y >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  fwrite(header, 1, 54, f);
  fwrite(buffer, 1, XSIZE * YSIZE * 3, f);
  fclose(f);
}

/* Given iteration number, set a colour */
void fancycolour(uchar *p, int iter) {
  if (iter == MAXITER);
  else if (iter < 8) { p[0] = 128 + iter * 16; p[1] = p[2] = 0; }
  else if (iter < 24) { p[0] = 255; p[1] = p[2] = (iter - 8) * 16; }
  else if (iter < 160) { p[0] = p[1] = 255 - (iter - 24) * 2; p[2] = 255; }
  else { p[0] = p[1] = (iter - 160) * 2; p[2] = 255 - (iter - 160) * 2; }
}

/* Create nice image from iteration counts. take care to create it upside down (bmp format) */
void output(){
    unsigned char *buffer = calloc(XSIZE * YSIZE * 3, 1);
    for (int i = 0; i < XSIZE; i++) {
      for (int j = 0; j < YSIZE; j++) {
        int p = ((YSIZE - j - 1) * XSIZE + i) * 3;
        fancycolour(buffer + p, pixel[(i + XSIZE * j)]);
      }
    }
    /* write image to disk */
    savebmp("mandel2.bmp", buffer, XSIZE, YSIZE);
    free(buffer);
}
