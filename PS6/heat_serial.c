#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>
#include <pthread.h>


/* Functions to be implemented: */
void ftcs_solver ( int step );
void external_heat ( int step );

/* Prototypes for functions found at the end of this file */
void write_temp ( int step );
void print_local_temps(int step);
void init_temp_material();
void init_local_temp();

/*
 * Physical quantities:
 * k                    : thermal conductivity      [Watt / (meter Kelvin)]
 * rho                  : density                   [kg / meter^3]
 * cp                   : specific heat capacity    [kJ / (kg Kelvin)]
 * rho * cp             : volumetric heat capacity  [Joule / (meter^3 Kelvin)]
 * alpha = k / (rho*cp) : thermal diffusivity       [meter^2 / second]
 *
 * Mercury:
 * cp = 0.140, rho = 13506, k = 8.69
 * alpha = 8.69 / (0.140*13506) =~ 0.0619
 *
 * Copper:
 * cp = 0.385, rho = 8960, k = 401
 * alpha = 401.0 / (0.385 * 8960) =~ 0.120
 *
 * Tin:
 * cp = 0.227, k = 67, rho = 7300
 * alpha = 67.0 / (0.227 * 7300) =~ 0.040
 *
 * Aluminium:
 * cp = 0.897, rho = 2700, k = 237
 * alpha = 237 / (0.897 * 2700) =~ 0.098
 */

const float MERCURY = 0.0619;
const float COPPER = 0.116;
const float TIN = 0.040;
const float ALUMINIUM = 0.098;


/* Size of the computational grid - 2048x2048 square */
const int GRID_SIZE[2] = {2048,2048};

/* Parameters of the simulation: how many steps, and when to cut off the heat */
const int NSTEPS = 10000;
const int CUTOFF = 5000;

/* How often to dump state to file (steps).
 */
const int SNAPSHOT = 500;


/* Arrays for the simulation data */
float
    *material,          // Global material constants, on rank 0
    *temperature[2];       // Global temperature field, on rank 0

/* Discretization: 5cm square cells, 2.5ms time intervals */
const float
    h  = 5e-2,
    dt = 2.5e-3;
    




/* Indexing functions, returns linear index for x and y coordinates, compensating for the border */

// temperature
int ti(int x, int y){
    
    if(x < 0){
        x++;
    }
    if(x >= GRID_SIZE[0]){
        x--;
    }
    if(y < 0){
        y++;
    }
    if(y >= GRID_SIZE[1]){
        y--;
    }
    
    return ((y)*(GRID_SIZE[0]) + x);
}

// material
int mi(int x, int y){
    return ((y)*(GRID_SIZE[0]) + x );
}



void ftcs_solver( int step ){
    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            float* in = temperature[(step)%2];
            float* out = temperature[(step+1)%2];
            
            out[ti(x,y)] = in[ti(x,y)] + material[mi(x,y)]*
                           (in[ti(x+1,y)] + 
                           in[ti(x-1,y)] + 
                           in[ti(x,y+1)] + 
                           in[ti(x,y-1)] -
                           4*in[ti(x,y)]);
        }
    }
}

void external_heat( int step ){
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            temperature[step%2][ti(x,y)] = 100.0;

        }
    }
}


    

int main ( int argc, char **argv ){
    
    size_t temperature_size =(GRID_SIZE[0])*(GRID_SIZE[1]);
    temperature[0] = calloc(temperature_size, sizeof(float));
    temperature[1] = calloc(temperature_size, sizeof(float));
    size_t material_size = (GRID_SIZE[0])*(GRID_SIZE[1]); 
    material = calloc(material_size, sizeof(float));
        
    init_temp_material();
    
        // Main integration loop: NSTEPS iterations, impose external heat
    for( int step=0; step<NSTEPS; step += 1 ){
        if( step < CUTOFF ){
            external_heat ( step );
        }
        ftcs_solver( step );
            
        if((step % SNAPSHOT) == 0){
            write_temp(step);
        }
    }
        
    free (temperature[0]);
    free (temperature[1]);
    free (material);
        
    exit ( EXIT_SUCCESS );
}




void init_temp_material(){
    
    for(int x = -(0); x < GRID_SIZE[0] + (0); x++){
        for(int y = -(0); y < GRID_SIZE[1] +(0); y++){
            temperature[0][ti(x,y)] = 10.0;
             temperature[1][ti(x,y)] = 10.0;

        }
    }
    
    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[0][ti(x,y)] = 20.0;
            material[mi(x,y)] = MERCURY * (dt/(h*h));
        }
    }
    
    /* Set up the two blocks of copper and tin */
    for(int x=(5*GRID_SIZE[0]/8); x<(7*GRID_SIZE[0]/8); x++ ){
        for(int y=(GRID_SIZE[1]/8); y<(3*GRID_SIZE[1]/8); y++ ){
            material[mi(x,y)] = COPPER * (dt/(h*h));
            temperature[0][ti(x,y)] = 60.0;
        }
    }
    
    for(int x=(GRID_SIZE[0]/8); x<(GRID_SIZE[0]/2)-(GRID_SIZE[0]/8); x++ ){
        for(int y=(5*GRID_SIZE[1]/8); y<(7*GRID_SIZE[1]/8); y++ ){
            
            material[mi(x,y)] = TIN * (dt/(h*h));
            temperature[0][ti(x,y)] = 60.0;
        }
    }

    /* Set up the heating element in the middle */
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            material[mi(x,y)] = ALUMINIUM * (dt/(h*h));
            temperature[0][ti(x,y)] = 100.0;
        }
    }
}


/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down */
void savebmp(char *name, unsigned char *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
  if (!f) {
    printf("Error writing image to disk.\n");
    return;
  }
  unsigned int size = x * y * 3 + 54;
  unsigned char header[54] = {'B', 'M',
                      size&255,
                      (size >> 8)&255,
                      (size >> 16)&255,
                      size >> 24,
                      0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x&255, x >> 8, 0,
                      0, y&255, y >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  fwrite(header, 1, 54, f);
  fwrite(buffer, 1, GRID_SIZE[0] * GRID_SIZE[1] * 3, f);
  fclose(f);
}

void fancycolour(unsigned char *p, float temp) {
    float r = (temp/101) * 255;
    
    if(temp <= 25){
        p[2] = 0;
        p[1] = (unsigned char)((temp/25)*255);
        p[0] = 255;
    }
    else if (temp <= 50){
        p[2] = 0;
        p[1] = 255;
        p[0] = 255 - (unsigned char)(((temp-25)/25) * 255);
    }
    else if (temp <= 75){
        
        p[2] = (unsigned char)(255* (temp-50)/25);
        p[1] = 255;
        p[0] = 0;
    }
    else{
        p[2] = 255;
        p[1] = 255 -(unsigned char)(255* (temp-75)/25) ;
        p[0] = 0;
    }
}

/* Create nice image from iteration counts. take care to create it upside down (bmp format) */
void output(char* filename, int step){
    unsigned char *buffer = calloc(GRID_SIZE[0] * GRID_SIZE[1]* 3, 1);
    for (int j = 0; j < GRID_SIZE[1]; j++) {
        for (int i = 0; i < GRID_SIZE[0]; i++) {
        int p = ((GRID_SIZE[1] - j - 1) * GRID_SIZE[0] + i) * 3;
        fancycolour(buffer + p, temperature[step%2][ti(i,j)]);
      }
    }
    /* write image to disk */
    savebmp(filename, buffer, GRID_SIZE[0], GRID_SIZE[1]);
    free(buffer);
}


void write_temp ( int step ){
    char filename[15];
    sprintf ( filename, "data/%.4d.bmp", step/SNAPSHOT );

    output ( filename, step );
    printf ( "Snapshot at step %d\n", step );
}
