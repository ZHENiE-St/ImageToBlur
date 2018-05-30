#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

/*
__global__ void separateChannels(const uchar4* const inputImageRGBA,
	int numRows,
	int numCols,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel)
{
	// TODO
	//
	// NOTE: Be careful not to try to access memory that is outside the bounds of
	// the image. You'll want code that performs the following check before accessing
	// GPU memory:
	//

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= numCols || py >= numRows) {
		return;
	}
	int i = py * numCols + px;
	redChannel[i] = inputImageRGBA[i].x;
	greenChannel[i] = inputImageRGBA[i].y;
	blueChannel[i] = inputImageRGBA[i].z;
}
*/

/*
__global__ void convertImage(float* inputChannel, float* outputChannel, int numRows, int numCols, float* filter, int filterWidth) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= numCols || py >= numRows) {
		return;
	}
	float c = 0.0f;

	for (int fx = 0; fx < filterWidth; fx++) {
		for (int fy = 0; fy < filterWidth; fy++) {
			int imagex = px + fx - filterWidth / 2;
			int imagey = py + fy - filterWidth / 2;
			imagex = min(max(imagex, 0), numCols - 1);
			imagey = min(max(imagey, 0), numRows - 1);
			c += (filter[fy*filterWidth + fx] * inputChannel[imagey*numCols + imagex]);
		}
	}
	outputChannel[py*numCols + px] = c;
}
/*
__global__ void convertImage(
	float *d_Result,
	float *d_Data,
	int dataW,
	int dataH)
{
		// global mem address for this thread
	const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * dataW + blockIdx.y * blockDim.y * dataW;
	float sum = 0;
	float value = 0;

	for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)	// row wise
		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)	// col wise
		{
		// check row first
			if (blockIdx.x == 0 && (threadIdx.x + i) < 0)	// left apron
			value = 0;
		else if (blockIdx.x == (gridDim.x - 1) && (threadIdx.x + i) > blockDim.x - 1)	// right apronvalue = 0;
		else{
		// check col next
			if (blockIdx.y == 0 && (threadIdx.y + j) < 0)	// top apronvalue = 0;
				else if (blockIdx.y == (gridDim.y - 1) &&
				(threadIdx.y + j) > blockDim.y - 1)	// bottom apron
				value = 0;
			else{	// safe case
				value = d_Data[gLoc + i + j * dataW];
			}
		sum += value * d_Kernel[KERNEL_RADIUS + i] * d_Kernel[KERNEL_RADIUS + j];
		}
	d_Result[gLoc] = sum;
}
*/

//@@ INSERT CODE HERE
/*
//zenith_coding...
__global__ void convertImage(float* input, float* output, int y, int x) {
	int tileidx = threadIdx.x;
	int tileidy = threadIdx.y;
	int xstart = max(0, blockIdx.x * blockDim.x);
	int ystart = max(0, blockIdx.y * blockDim.y);
	int xend = min(int columns, xstart + blockDim.x + 2 * radius);
	int yend = min(int columns, ystart + blockDim.y + 2 * radius);
	for (int y = 0; ystart + threadIdx.y <= yend − 1; blockDim.y++) {
		for (int x = 0; xstart + threadIdx.x <= xend - 1; blockDim.x++) {
			shared[tileidx + tileidy * tiledim] = inputImage[x + y * columns];
			tileidx = tileidx + blockDim.x;
		}
		tileidy = tileidy + blockDim.y;
		tileidx = threadIdx.x;
	}
	synchronize();

}
*/
/*
__global__ void convertImage(float* input , float* output , int width, int height) {
	for (int i = 0; i <= height; i++) {
		for (int j = 0; j <= width; j++) {
			int pos = i*width + j;
			float c = 0.0f;
			float r = input[3 * pos];
			float g = input[3 * pos+1];
			float b = input[3 * pos+2];
			output[pos] = (0.21*r) + (0.71*g) + (0.07*b);
		}
	}
}
*/

__global__ void convertImage(float* in, float* out, int width, int height, int blur_size) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;


	if (col < width && row < height) {
		int pixVal = 0;
		int pixels = 0;
		for (int blurRow = -blur_size; blurRow < blur_size + 1; ++blurRow) {
			for (int blurCol = -blur_size; blurCol < blur_size + 1; ++blurCol) {
				int curRow = row + blurRow;
				int curCol = col + blurCol;
				if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
					pixVal += in[curRow*width + curCol];
					pixels++;
				}
			}
		}
		out[row*width + col] = (unsigned char)(pixVal / pixels);
	}
}
int main(int argc, char *argv[]) {
	wbArg_t args;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	int boxFilter = 5;
	int blur_Range = boxFilter / 2;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);
	inputImage = wbImport(inputImageFile);

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);


	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	//wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	//wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,
		imageWidth * imageHeight * sizeof(float));
	//wbTime_stop(GPU, "Doing GPU memory allocation");
	convertImage << <1, 512 >> > (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, blur_Range);
	//wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float),cudaMemcpyHostToDevice);
	//wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	//wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	
	//wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	//wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * sizeof(float),
		cudaMemcpyDeviceToHost);
	//wbTime_stop(Copy, "Copying data from the GPU");

	//wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}