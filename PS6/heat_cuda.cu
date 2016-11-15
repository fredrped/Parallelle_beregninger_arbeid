#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

/* Functions to be implemented: */
float ftcs_solver_gpu ( int step, int block_size_x, int block_size_y );
float ftcs_solver_gpu_shared ( int step, int block_size_x, int block_size_y );
float ftcs_solver_gpu_texture ( int step, int block_size_x, int block_size_y );
void external_heat_gpu ( int step, int block_size_x, int block_size_y );
void transfer_from_gpu( int step );
void transfer_to_gpu();
void device_allocation();
//int ti(int,int)
/* Prototypes for functions found at the end of this file */
void write_temp( int step );
void print_local_temps();
void init_temp_material();
void init_local_temp();
void host_allocation();
void add_time(float time);
void print_time_stats();

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

/* Discretization: 5cm square cells, 2.5ms time intervals */
const float
    h  = 5e-2,
    dt = 2.5e-3;

/* Size of the computational grid - 1024x1024 square */
const int GRID_SIZE[2] = {1024, 1024};
//__constant__ int GRID_SIZE_DEVICE[2] = {2048, 2048};
__device__ int GRID_SIZE_DEVICE[2] = {1024, 1024};
/* Parameters of the simulation: how many steps, and when to cut off the heat */
const int NSTEPS = 10000;
const int CUTOFF = 5000;

/* How often to dump state to file (steps). */
const int SNAPSHOT = 500;

/* For time statistics */
float min_time = -2.0;
float max_time = -2.0;
float avg_time = 0.0;

/* Arrays for the simulation data, on host */
float
    *material,          // Material constants
    *temperature;       // Temperature field

/* Arrays for the simulation data, on device */
float
    *material_device,           // Material constants
    *temperature_device[2];      // Temperature field, 2 arrays 

texture<float, cudaTextureType2D> temperature_device_ref;



int iDivup(int a, int b){return ((a%b)!=0) ? (a/b + 1) : (a/b);  }  


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



/* Allocate arrays on GPU */
void device_allocation(){
	
	cudaMalloc((void**)&temperature_device[0],GRID_SIZE[0]*GRID_SIZE[1]*sizeof(float)); //Is this correct regarding temp, since it has dim=2

	cudaMalloc((void**)&temperature_device[1],GRID_SIZE[0]*GRID_SIZE[1]*sizeof(float)); //Is this correct regarding temp, since it has dim=2
	cudaMalloc((void**)&material_device,GRID_SIZE[0]*GRID_SIZE[1]*sizeof(float)); 
}

//void cpuAlloc( void ** ptr, size_t size){
//	*ptr=calloc(size, 1);
//}

/* Transfer input to GPU */
void transfer_to_gpu(){
	cudaMemcpy(temperature_device[0], temperature, GRID_SIZE[0]*GRID_SIZE[1]*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(material_device, material, GRID_SIZE[0]*GRID_SIZE[1]*sizeof(float),cudaMemcpyHostToDevice);
}

/* Transfer output from GPU to CPU */
void transfer_from_gpu(int step){//unsure of the index?
	cudaMemcpy(temperature,temperature_device[step%2],GRID_SIZE[0]*GRID_SIZE[1]*sizeof(float),cudaMemcpyDeviceToHost);
	//cudaMemcpy(material,material_device,GRID_SIZE[0]*GRID_SIZE[1]*sizeof(float),cudaMemcpyDeviceToHost);
}



__device__ int ti(int x, int y){

    if(x < 0){
        x++;
    }
    if(x >= GRID_SIZE_DEVICE[0]){
        x--;
    } 
    if(y < 0){
        y++;
    }
    if(y >= GRID_SIZE_DEVICE[1]){
        y--;
    }

    return ((y)*(GRID_SIZE_DEVICE[0]) + x);
}


/* Plain/global memory only kernel*/
__global__ void  ftcs_kernel( float* in, float* out, float* material_map ){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	//float* in = heat_map[step%2];
	//float* out = heat_map[(step+1)%2];
	if ((x<(GRID_SIZE_DEVICE[0])) &&( y<(GRID_SIZE_DEVICE[1]))){
		out[ti(x,y)] = in[ti(x,y)] + material_map[ ti(x,y)]*
				(in[ti(x+1,y)] +
				in[ti(x-1,y)] +
				in[ti(x,y+1)] +
				in[ti(x,y-1)] -
				4*in[ti(x,y)]);
		}

} 





/* Shared memory kernel */
__global__ void  ftcs_kernel_shared( float* in, float* out, float* material_map, int GLOBAL_GRID_SIZE_X ){
	extern __shared__ float tile[];
	//gives weird results if blockDim is odde

	int global_x = blockIdx.x*blockDim.x + threadIdx.x;
	int global_y = blockIdx.y*blockDim.y + threadIdx.y;

	int local_x = threadIdx.x;
	int local_y = threadIdx.y;

	int local_pos = local_x + local_y*blockDim.x;

	int local_pos_w = local_x-1 + local_y*blockDim.x;
	int local_pos_e = local_x+1 + local_y*blockDim.x;
	int local_pos_n = local_x + (local_y-1)*blockDim.x;
	int local_pos_s = local_x + (local_y+1)*blockDim.x;
	
	float west,east,north,south;
	//maping in array to shared array
	if ((global_x<(GRID_SIZE_DEVICE[0])) &&( global_y<(GRID_SIZE_DEVICE[1]))){
//	__syncthreads();
		tile[local_pos] = in[global_x+global_y*GLOBAL_GRID_SIZE_X];

		__syncthreads();

		if(local_x==0){
			west = in[ti(global_x-1,global_y)]; 
		
		}
		else{
			west = tile[local_pos_w];
		}
		if((local_x==(blockDim.x-1))||(global_x==GRID_SIZE_DEVICE[0]-1)){
			east = in[ti(global_x+1,global_y)];
		 
		}
		else{
			east = tile[local_pos_e];
		}
		if(local_y==0){
			north = in[ti(global_x,global_y-1)];
		
		}
		else{
			north = tile[local_pos_n];
		}
		if(local_y==(blockDim.y-1)||(global_y==GRID_SIZE_DEVICE[1]-1)){
			south = in[ti(global_x,global_y+1)];
		
		}
		else{
			south = tile[local_pos_s];
		}
	

		out[ti(global_x,global_y)] = tile[local_pos] + material_map[ti(global_x,global_y)]*
				(west +
				east +
				north +
				south -
				4*tile[local_pos]);

	}
}




/* Texture memory kernel */
__global__ void  ftcs_kernel_texture(float* out, float* material_map /* Add arguments here */ ){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y; 
	if ((x<(GRID_SIZE_DEVICE[0])) &&( y<(GRID_SIZE_DEVICE[1]))){

	out[ti(x,y)] = tex2D(temperature_device_ref, x, y) + material_map[ ti(x,y)]*
                                (tex2D(temperature_device_ref, x+1, y) +
                                 tex2D(temperature_device_ref, x-1, y) +
                                 tex2D(temperature_device_ref, x, y+1) +
                                 tex2D(temperature_device_ref, x, y-1) -
                               4*tex2D(temperature_device_ref, x, y));

	}
}

/* External heat kernel, should do the same work as the external
 * heat function in the serial code 
 */
__global__ void external_heat_kernel( float* heat_map, int x_0, int y_0, int GRID_SIZE_X){
	
	int x = blockIdx.x*blockDim.x + threadIdx.x + x_0;
	int y = blockIdx.y*blockDim.y + threadIdx.y + y_0;//ask questions here how does this work
	if ((x<(GRID_SIZE_DEVICE[0])) &&( y<(GRID_SIZE_DEVICE[1]))){
		heat_map[y*GRID_SIZE_X+x]=100;	
	}
}
/* Set up and call ftcs_kernel
 * should return the execution time of the kernel
 */
float ftcs_solver_gpu( int step, int block_size_x, int block_size_y ){
	int num_block_x_direc = iDivup(GRID_SIZE[0],block_size_x);
	int num_block_y_direc = iDivup(GRID_SIZE[1],block_size_y);
	
	dim3 gridBlock(num_block_x_direc, num_block_y_direc);
        dim3 threadBlock(block_size_x,block_size_y);

	float time = 0;	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	gpuErrchk(cudaPeekAtLastError());
	
	cudaEventRecord(start);	
	ftcs_kernel<<<gridBlock,threadBlock>>>( temperature_device[step%2], temperature_device[(step+1)%2], material_device);
    	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
    return time;
}




/* Set up and call ftcs_kernel_shared
 * should return the execution time of the kernel
 */
float ftcs_solver_gpu_shared( int step, int block_size_x, int block_size_y ){
	//THE QUESTION IS WHERE TO ALLOCATE THE SHARED MEMORY
	int num_block_x_direc = iDivup(GRID_SIZE[0],block_size_x);
        int num_block_y_direc = iDivup(GRID_SIZE[1],block_size_y);

	size_t tile_size = block_size_x*block_size_y*sizeof(float);
        dim3 gridBlock(num_block_x_direc, num_block_y_direc);
    	dim3 threadBlock(block_size_x,block_size_y);
	
	//remember this do not include the halo. should i allocate share mem here instead?
	float time = 0;	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	ftcs_kernel_shared<<<gridBlock,threadBlock, tile_size>>>(temperature_device[step%2], temperature_device[(step+1)%2], material_device, GRID_SIZE[0] );

	cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time,start,stop);    	
    return time;
}




/* Set up and call ftcs_kernel_texture
 * should return the execution time of the kernel
 */
float ftcs_solver_gpu_texture( int step, int block_size_x, int block_size_y ){
	
	int num_block_x_direc = iDivup(GRID_SIZE[0],block_size_x);
        int num_block_y_direc = iDivup(GRID_SIZE[1],block_size_y);

    	
	dim3 gridBlock(num_block_x_direc, num_block_y_direc);
        dim3 threadBlock(block_size_x,block_size_y);
        gpuErrchk(cudaPeekAtLastError());

	float time = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

	//binding texture array to input array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();//these two are wrong
	cudaBindTexture2D(NULL, temperature_device_ref, temperature_device[step%2], channelDesc, GRID_SIZE[0],GRID_SIZE[1],GRID_SIZE[0]*sizeof(float));
 	
	cudaEventRecord(start);   	
	
	ftcs_kernel_texture<<<gridBlock,threadBlock>>>( temperature_device[(step+1)%2], material_device);//thought i had to send to kernel?
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	
    	return time;
}


/* Set up and call external_heat_kernel improvement is to have threadwarp fully utilized*/
void external_heat_gpu( int step, int block_size_x, int block_size_y){
	//use ceil to make sure i allocate enough threads
	int num_block_x_direc = iDivup(GRID_SIZE[0],(2*block_size_x));
	int num_block_y_direc = iDivup(GRID_SIZE[1],(8*block_size_y));
	//this is not complete
	gpuErrchk(cudaPeekAtLastError());
	dim3 gridBlock(num_block_x_direc, num_block_y_direc);
        dim3 threadBlock(block_size_x,block_size_y); //need to check dimensions to be consistent 
        external_heat_kernel<<<gridBlock,threadBlock>>>(temperature_device[step%2], GRID_SIZE[0]/4, GRID_SIZE[1]/2-GRID_SIZE[1]/16, GRID_SIZE[0] );//curios on the notation
	//need to do more not sure what do do next??? 
}
//test
void print_gpu_info(){
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  printf("Number of CUDA devices: %d\n", n_devices);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  printf("CUDA device name: %s\n" , device_prop.name);
  printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
}


int main ( int argc, char **argv ){
    
    // Parse command line arguments
    int version = 0;
    int block_size_x = 0;
    int block_size_y = 0;
    if(argc != 4){
        printf("Useage: %s <version> <block_size_x> <block_size_y>\n\n<version> can be:\n0: plain\n1: shared memory\n2: texture memory\n", argv[0]);
        exit(0);
    }
    else{
        version = atoi(argv[1]);
        block_size_x = atoi(argv[2]);
        block_size_y = atoi(argv[3]);
    }
    
    print_gpu_info();
    
    // Allocate and initialize data on host
    host_allocation();
    init_temp_material();
   //test 
    // Allocate arrays on device, and transfer inputs
    device_allocation();
    transfer_to_gpu();
        
    // Main integration loop
    for( int step=0; step<NSTEPS; step += 1 ){
        
        if( step < CUTOFF ){
            external_heat_gpu ( step, block_size_x, block_size_y );
        }
        
        float time;
        // Call selected version of ftcs slover
        if(version == 2){
            time = ftcs_solver_gpu_texture( step, block_size_x, block_size_y );
        }
        else if(version == 1){
            time = ftcs_solver_gpu_shared(step, block_size_x, block_size_y);
        }
        else{
            time = ftcs_solver_gpu(step, block_size_x, block_size_y);
        }
        
        add_time(time);
        
        if((step % SNAPSHOT) == 0){
            // Transfer output from device, and write to file
            transfer_from_gpu(step);
            write_temp(step);
        }
    }
    
    print_time_stats();
        
    exit ( EXIT_SUCCESS );
}


void host_allocation(){
    size_t temperature_size =GRID_SIZE[0]*GRID_SIZE[1];
    temperature = (float*) calloc(temperature_size, sizeof(float));
    size_t material_size = (GRID_SIZE[0])*(GRID_SIZE[1]); 
    material = (float*) calloc(material_size, sizeof(float));
}


void init_temp_material(){
    
    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[y * GRID_SIZE[0] + x] = 10.0;

        }
    }
    
    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[y * GRID_SIZE[0] + x] = 20.0;
            material[y * GRID_SIZE[0] + x] = MERCURY * (dt/(h*h));
        }
    }
    
    /* Set up the two blocks of copper and tin */
    for(int x=(5*GRID_SIZE[0]/8); x<(7*GRID_SIZE[0]/8); x++ ){
        for(int y=(GRID_SIZE[1]/8); y<(3*GRID_SIZE[1]/8); y++ ){
            material[y * GRID_SIZE[0] + x] = COPPER * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 60.0;
        }
    }
    
    for(int x=(GRID_SIZE[0]/8); x<(GRID_SIZE[0]/2)-(GRID_SIZE[0]/8); x++ ){
        for(int y=(5*GRID_SIZE[1]/8); y<(7*GRID_SIZE[1]/8); y++ ){
            material[y * GRID_SIZE[0] + x] = TIN * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 60.0;
        }
    }

    /* Set up the heating element in the middle */
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            material[y * GRID_SIZE[0] + x] = ALUMINIUM * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 100.0;
        }
    }
}


void add_time(float time){
    avg_time += time;
    
    if(time < min_time || min_time < -1.0){
        min_time = time;
    }
    
    if(time > max_time){
        max_time = time;
    }
}

void print_time_stats(){
    printf("Kernel execution time (min, max, avg): %f %f %f\n", min_time, max_time, avg_time/NSTEPS);
}

/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down
 * Only works for images which dimensions are powers of two
 */
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
void output(char* filename){
    unsigned char *buffer = (unsigned char*)calloc(GRID_SIZE[0] * GRID_SIZE[1]* 3, 1);
    for (int j = 0; j < GRID_SIZE[1]; j++) {
        for (int i = 0; i < GRID_SIZE[0]; i++) {
        int p = ((GRID_SIZE[1] - j - 1) * GRID_SIZE[0] + i) * 3;
        fancycolour(buffer + p, temperature[j*GRID_SIZE[0] + i]);
      }
    }
    /* write image to disk */
    savebmp(filename, buffer, GRID_SIZE[0], GRID_SIZE[1]);
    free(buffer);
}


void write_temp (int step ){
    char filename[15];
    sprintf ( filename, "data/%.4d.bmp", step/SNAPSHOT );

    output ( filename );
    printf ( "Snapshot at step %d\n", step );
}
