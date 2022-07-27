#include <iostream>
#include <math.h>
#include <cuda_profiler_api.h>
#include <cstdint>

#include "lodepng.h"
#include <raylib.h>

#define SENSOR_DIST_FROM_AGENT 5
#define SENSOR_SIZE_FROM_AGENT 5
#define DIFFUSE_RADIUS 4
#define EVAPORATION_SPEED 0.75
#define DIFFUSE_SPEED 50
#define TRAIL_STRENGTH 0.125

struct agent{
	float xPos, yPos;
	float rotation;
	float speed;
};

__device__
float lerp(float a, float b, float t){
	return a + t * (b - a);
}

__global__
void updateAgents(int n, agent* agents, float* floatImage, int imageWidth, int imageHeight, float delatTime){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < n; i += stride){
		float sense0sum, sense1sum, sense2sum;
		int sensor0xPos = floor(agents[i].xPos + (sin(agents[i].rotation * DEG2RAD + 45 * DEG2RAD) *
				SENSOR_DIST_FROM_AGENT));
		int sensor0yPos = floor(agents[i].yPos + (cos(agents[i].rotation * DEG2RAD + 45 * DEG2RAD) *
				SENSOR_DIST_FROM_AGENT));
		int sensor1xPos = floor(agents[i].xPos + (sin(agents[i].rotation * DEG2RAD) * SENSOR_DIST_FROM_AGENT));
		int sensor1yPos = floor(agents[i].yPos + (cos(agents[i].rotation * DEG2RAD) * SENSOR_DIST_FROM_AGENT));
		int sensor2xPos = floor(agents[i].xPos + (sin(agents[i].rotation * DEG2RAD - 45 * DEG2RAD) *
				SENSOR_DIST_FROM_AGENT));
		int sensor2yPos = floor(agents[i].yPos + (cos(agents[i].rotation * DEG2RAD - 45 * DEG2RAD) *
				SENSOR_DIST_FROM_AGENT));

		for(int x = -SENSOR_SIZE_FROM_AGENT; x < SENSOR_SIZE_FROM_AGENT; x++){
			for(int y = -SENSOR_SIZE_FROM_AGENT; y < SENSOR_SIZE_FROM_AGENT; y++){
				int s0px = sensor0xPos + x;
				int s0py = sensor0yPos + y;
				int s1px = sensor1xPos + x;
				int s1py = sensor1yPos + y;
				int s2px = sensor2xPos + x;
				int s2py = sensor2yPos + y;

				if(s0px >= 0 && s0px < imageWidth && s0py >= 0 && s0py < imageHeight)
					sense0sum += floatImage[s0px + s0py * imageWidth];
				if(s1px >= 0 && s1px < imageWidth && s1py >= 0 && s1py < imageHeight)
					sense1sum += floatImage[s1px + s1py * imageWidth];
				if(s2px >= 0 && s2px < imageWidth && s2py >= 0 && s2py < imageHeight)
					sense2sum += floatImage[s2px + s2py * imageWidth];

			}
		}

		if(sense0sum > sense1sum && sense0sum > sense2sum)
			agents[i].rotation += 360 * delatTime;
		else if(sense2sum > sense1sum && sense2sum > sense0sum)
			agents[i].rotation -= 360 * delatTime;

		agents[i].xPos += sin(agents[i].rotation * DEG2RAD) * agents[i].speed * delatTime;
		agents[i].yPos += cos(agents[i].rotation * DEG2RAD) * agents[i].speed * delatTime;

		if(agents[i].xPos <= 0 || agents[i].xPos >= imageWidth || agents[i].yPos <= 0 || agents[i].yPos >= imageHeight)
		{
			agents[i].rotation += 180;
			agents[i].xPos = max(0.0f, min(agents[i].xPos, imageWidth - 1.0f));
			agents[i].yPos = max(0.0f, min(agents[i].yPos, imageWidth - 1.0f));
		}
	}
}

__global__
void leaveTrail(int n, agent* agents, float* floatImage, int imageWidth, int imageHeight){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < n; i += stride){
		int x = floor(agents[i].xPos);
		int y = floor(agents[i].yPos);
		floatImage[x + y * imageWidth] = max(0.0f, min(1.0f, floatImage[x + y * imageWidth] + TRAIL_STRENGTH));
	}
}

__global__
void diffuseTrail(float* currentFloatImage, float* newFloatImage, int imageSize, int imageWidth, float deltaTime){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0;
	int hits = 0;

	for(int dx = x - 1; dx <= x + 1; dx++)
		for(int dy = y - 1; dy <= y + 1; dy++)
			if(dx >= 0 && dx < imageWidth && dy >= 0 && dy < imageSize / imageWidth)
				sum += currentFloatImage[dx + dy * imageWidth]; hits++;
	float blurResult = sum / 9.0f;

	float diffuseWeight = saturate(DIFFUSE_SPEED * deltaTime);

	float diffusedValue = currentFloatImage[x + y * imageWidth] * (1 - diffuseWeight) + blurResult * diffuseWeight;

	newFloatImage[x + y * imageWidth] = max(0.0f, diffusedValue - EVAPORATION_SPEED * deltaTime);
}

__global__
void floatImageToRGBAByteImage(float* floatImage, uint8_t* rgbaImage, int imageSize, int imageWidth){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = x * 4 + y * imageWidth * 4;
	int floatIndex = x + y * imageWidth;
	rgbaImage[index] = (floatImage[floatIndex] * 255);
	rgbaImage[index + 1] = (floatImage[floatIndex] * 255);
	rgbaImage[index + 2] = (floatImage[floatIndex] * 255);
	rgbaImage[index + 3] = 255;
}

int main()
{
	std::string filePath = "./frames/";
	int numAgents = 500000;
	float* currentFloatImage;
	float* newFloatImage;
	uint8_t* gpuRGBAImage;
	uint8_t* hostRGBAImage;
	agent* gpuAgents;
	agent* hostAgents;
	int imageWidth = 1920;
	int imageHeight = 1080;
	int frameAmount = 250;

	system("mkdir -p ./frames");

	hostRGBAImage = (uint8_t*)malloc(imageWidth * imageHeight * 4);
	hostAgents = (agent*)malloc(numAgents * sizeof(agent));

	cudaMalloc(&gpuRGBAImage, imageWidth * imageHeight * 4);
	cudaMalloc(&gpuAgents, numAgents * sizeof(agent));
	cudaMalloc(&currentFloatImage, imageWidth * imageHeight * sizeof(float));
	cudaMalloc(&newFloatImage, imageWidth * imageHeight * sizeof(float));

	cudaMemset(currentFloatImage, 0, imageWidth * imageHeight * sizeof(float));
	cudaMemset(newFloatImage, 0, imageWidth * imageHeight * sizeof(float));

	for(int i = 0; i < numAgents; i++){
		hostAgents[i].xPos = imageWidth / 2 + (rand() % 600 - 300);
		hostAgents[i].yPos = imageHeight / 2 + (rand() % 600 - 300);
		//hostAgents[i].xPos = imageWidth / 2;
		//hostAgents[i].yPos = imageHeight / 2;
		hostAgents[i].rotation = atan2(imageWidth / 2 - hostAgents[i].xPos, imageHeight / 2 - hostAgents[i].yPos) - 90;
		hostAgents[i].speed = 50;
	}

	cudaMemcpy(gpuAgents, hostAgents, numAgents * sizeof(agent), cudaMemcpyHostToDevice);

	/*
	for(int frame = 0; frame < frameAmount; frame++)
	{
		std::cout << "rendering image no:" << frame << std::endl;
		int blockSize = 256;
		int gridSize = ceil(numAgents / blockSize) + 1;

		updateAgents<<<gridSize, blockSize>>>(numAgents, gpuAgents, currentFloatImage, imageWidth, imageHeight);
		cudaDeviceSynchronize();

		leaveTrail<<<gridSize, blockSize>>>(numAgents, gpuAgents, currentFloatImage, imageWidth, imageHeight);
		cudaDeviceSynchronize();

		dim3 blockDim(32, 32);
		dim3 gridDim(ceil(imageWidth / blockDim.x), ceil(imageHeight / blockDim.y) + 1);

		diffuseTrail<<<gridDim, blockDim>>>(currentFloatImage, newFloatImage, imageWidth * imageHeight, imageWidth);
		cudaDeviceSynchronize();

		float* temp = currentFloatImage;
		currentFloatImage = newFloatImage;
		newFloatImage = temp;

		floatImageToRGBAByteImage<<<gridDim, blockDim>>>(currentFloatImage, gpuRGBAImage, imageWidth * imageHeight,
		                                                 imageWidth);
		cudaDeviceSynchronize();

		cudaMemcpy(hostRGBAImage, gpuRGBAImage, imageWidth * imageHeight * 4, cudaMemcpyDeviceToHost);

		//lodepng::encode(filePath + std::to_string(frame) + ".png", hostRGBAImage, imageWidth, imageHeight);
	}
	 */

	InitWindow(1920, 1080, "Slime Mold Simulation");
	Image image = GenImageColor(imageWidth, imageHeight, RAYWHITE);

	Texture2D texture = LoadTextureFromImage(image);

	// raylib
	while(!WindowShouldClose())
	{
		int blockSize = 512;
		int gridSize = (numAgents + blockSize - 1) / blockSize;

		updateAgents<<<gridSize, blockSize>>>(numAgents, gpuAgents, currentFloatImage, imageWidth, imageHeight,
											  GetFrameTime());
		cudaDeviceSynchronize();

		leaveTrail<<<gridSize, blockSize>>>(numAgents, gpuAgents, currentFloatImage, imageWidth, imageHeight);
		cudaDeviceSynchronize();

		dim3 blockDim(32, 32);
		dim3 gridDim(ceil(imageWidth / blockDim.x), ceil(imageHeight / blockDim.y) + 1);

		diffuseTrail<<<gridDim, blockDim>>>(currentFloatImage, newFloatImage, imageWidth * imageHeight, imageWidth,
											GetFrameTime());
		cudaDeviceSynchronize();

		float* temp = currentFloatImage;
		currentFloatImage = newFloatImage;
		newFloatImage = temp;

		floatImageToRGBAByteImage<<<gridDim, blockDim>>>(currentFloatImage, gpuRGBAImage, imageWidth * imageHeight,
		                                                 imageWidth);
		cudaDeviceSynchronize();

		cudaMemcpy(hostRGBAImage, gpuRGBAImage, imageWidth * imageHeight * 4, cudaMemcpyDeviceToHost);

		UpdateTexture(texture, hostRGBAImage);

		BeginDrawing();
		ClearBackground(RAYWHITE);
		DrawTexture(texture, 0, 0, WHITE);
		DrawFPS(100, 100);
		EndDrawing();
	}

	CloseWindow();

	cudaFree(gpuAgents);
	cudaFree(gpuRGBAImage);
	cudaFree(currentFloatImage);
	cudaFree(newFloatImage);

	free(hostAgents);
	free(hostRGBAImage);
	return 0;
}
