#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <array>
#include<SDL3/SDL.h>
#include <sstream> 
#include <iostream>
#include "util.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "aabb.h"
#include "bvh.h"
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}



__device__ Vec3 color(const Ray& r,BVH_NODE** world,curandState* state)
{

    Vec3 accColor(1.0, 1.0, 1.0);
    Ray currentRay = r;
    curandState local_state = *state;
    for (int i = 0; i < 50; i++)
    {
        HitRecord rec;
        if ( (*world)->hit(currentRay,0.001f,FLT_MAX,rec))
        {
            Ray scattered;
            Vec3 attenuation;
            if (rec.material->Scatter(currentRay, rec, attenuation, scattered, &local_state))
            {
                accColor *= attenuation;
                currentRay = scattered;
            }
            else {
                return accColor;
            }   
        }
        else {
            Vec3 unit_direction = unit_vector(currentRay.direction());
            float t = 0.1f * (unit_direction.y() + 1.0f);
            *state = local_state;
            return accColor * (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
        }
    }
    *state = local_state;
    return Vec3(0.0, 0.0, 0.0);
}


__global__ void render_init(int width,int height,curandState* state)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;
    int index_pixel = y * width + x;
    curand_init(1984 + index_pixel, index_pixel, 0, &state[index_pixel]);
}

__global__ void render(uint32_t* fb, int width, int height,BVH_NODE** world,Camera** camera,curandState* state,int nr)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;
    int pixel_index = (height - 1 - j) * width  + i;
    curandState local_state = state[pixel_index];
    Vec3 col(0.0, 0.0, 0.0);
    for (int sample = 0; sample < nr; sample++)
    {
        float u = (float(i) + curand_uniform(&local_state)) / float(width);
        float v = (float(j) + curand_uniform(&local_state)) / float(height);
        Ray r = (*camera)->getRay(u, v);
        col += color(r, world,&local_state);
    }
    state[pixel_index] = local_state;
    col /= double(nr);
    col[0] = sqrt(col[0]) * 255;
    col[1] = sqrt(col[1]) * 255;
    col[2] = sqrt(col[2]) * 255;
    fb[pixel_index] = (255 <<24) | (uint8_t(col[0]) << 16) |(uint8_t(col[1])<< 8) | uint8_t(col[2]);
}

__global__ void createWorld(Hittable** list, BVH_NODE** world,Camera** camera,int nx,int ny,curandState* state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        list[0] = new Sphere(Vec3(0.0, 0.0, -2.2), 0.5, new Lambertian(Vec3(1.0, 0.0, 0.0)));
        list[1] = new Sphere(Vec3(0.0, -100.5, -2.0),100, new Lambertian(Vec3(0.0,0.0,1.0)));
        list[2]= new Sphere(Vec3(-1.0, 0.0, -2.0), 0.5, new Dielectric(1.00/1.33));
        list[3] = new Sphere(Vec3(1.0, 0.0, -2.0), 0.5, new Metal(Vec3(0.3, 0.5, 0.4),0.5));
        *world = new BVH_NODE(list, 0,4,state);
        *camera = new Camera(Vec3(-2.0, 1.0, 1.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0.0 , 1.0, 0.0),
            90.0,
            float(nx) / float(ny),state);
    }
}
__global__ void clearWorld(Hittable** list, BVH_NODE** world,Camera** camera)
{
    for (int i = 0; i < 4; i++)
    {
        delete list[i];
    }
    delete *world;
    delete* camera;
}
int main()
{   
    SDL_Window* window;
    SDL_Renderer* renderer;
    cudaDeviceProp deviceProps;

    cudaGetDeviceProperties(&deviceProps, 0);
    cudaDeviceSetLimit(cudaLimitStackSize, 10240);
    bool running = true;
    int width = 1240;
    int height = 720;
    int numPixel = width * height;
    size_t size = numPixel * sizeof(uint32_t);
    int tx = 16;
    int ty = 16;
    int nr = 100;

    uint32_t* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, size));

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("erro ao inicialziar window: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    window = SDL_CreateWindow("RayTracing_cuda", width, height,SDL_EVENT_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, nullptr);
    if (!window || !renderer) {
        std::cerr << "Erro ao criar janela ou renderizador: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Texture* textura = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!textura) {
        std::cerr << "Erro ao criar textura: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_SetRenderDrawColor(renderer, 255, 255, 255,255);
    SDL_RenderClear(renderer);

    Hittable** list;
    checkCudaErrors(cudaMalloc((void**)&list, 4 * sizeof(Hittable*)));

    BVH_NODE** world;
    checkCudaErrors(cudaMalloc((void**)&world, sizeof(BVH_NODE*)));
    Camera** camera;
    checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));
    curandState* rand_state;
    checkCudaErrors(cudaMalloc((void**)&rand_state,numPixel * sizeof(curandState)));

    createWorld << <1, 1 >> > (list, world,camera,16.0,9.0,rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(ceil(width/ (float)tx),ceil( height/ (float)ty));
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (width, height, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render <<<blocks, threads >>> (fb, width, height,world,camera,rand_state,nr);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    Uint64 lastTime = SDL_GetPerformanceCounter();
    Uint64 currentTime;
    double deltaTime;
    int frameCount = 0;
    double elapsedTime = 0;
    std::string fpsText = "FPS: 0";
   

    while (running)
    {
        currentTime = SDL_GetPerformanceCounter();
        deltaTime = (double)(currentTime - lastTime) / SDL_GetPerformanceFrequency(); 
        lastTime = currentTime;

        elapsedTime += deltaTime;
        frameCount++;
        
        SDL_Event event;
        while (SDL_PollEvent(&event) != 0)
        {
            if (event.type == SDL_EVENT_QUIT)
            {
                running = false;
            }
           
            if (elapsedTime >= 1.0) {  
                std::ostringstream fpsStream;
                fpsStream <<deviceProps.name << " FPS: " << static_cast<int>(frameCount / elapsedTime);
                fpsText = fpsStream.str();
                SDL_SetWindowTitle(window,fpsText.c_str());
                frameCount = 0;
                elapsedTime = 0;
            }
            SDL_UpdateTexture(textura, nullptr, fb, width * sizeof(uint32_t));
            SDL_RenderClear(renderer);
            SDL_RenderTexture(renderer, textura, nullptr, nullptr);
            SDL_RenderPresent(renderer);
        }
    }


    SDL_DestroyTexture(textura);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    checkCudaErrors(cudaDeviceSynchronize());
    clearWorld<<<1,1>>>(list, world,camera);
    checkCudaErrors(cudaGetLastError());
    cudaFree(rand_state);
    cudaFree(camera);
    cudaFree(list);
    cudaFree(world);
    cudaFree(fb);
}
