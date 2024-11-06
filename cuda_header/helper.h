#include <iostream>
#include <atomic>
#include <stdexcept>
#include "utils/log.h"
#include <stdio.h>
#include <cuda_fp16.h>
#include "cutil.h"



inline __device__ float2 RayAABBIntersection(
  const float3 ori,
  const float3 dir,
  const float3 center,
  float3 half_size) {

  float f_low = 0;
  float f_high = 100000.;
  float f_dim_low, f_dim_high, temp, inv_ray_dir, start, aabb, half_voxel;

  for (int d = 0; d < 3; ++d) {  
    switch (d) {
      case 0:
        inv_ray_dir = safe_divide(1.0f, dir.x); start = ori.x; aabb = center.x; half_voxel = half_size.x; break;
      case 1:
        inv_ray_dir = safe_divide(1.0f, dir.y); start = ori.y; aabb = center.y; half_voxel = half_size.y; break;
      case 2:
        inv_ray_dir = safe_divide(1.0f, dir.z); start = ori.z; aabb = center.z; half_voxel = half_size.z; break;
    }
    
    //     0.35       (0 - 1 - -1.35) * 1
    f_dim_low  = (aabb - half_voxel - start) * inv_ray_dir;
    //      4.48       (0 + 1 - -0.12) * 4
    f_dim_high = (aabb + half_voxel - start) * inv_ray_dir;
  
    // Make sure low is less than high
    if (f_dim_high < f_dim_low) {
      temp = f_dim_low;
      f_dim_low = f_dim_high;
      f_dim_high = temp;
    }

    // If this dimension's high is less than the low we got then we definitely missed.
    if (f_dim_high < f_low) {
      return make_float2(-1.0f, -1.0f);
    }
  
    // Likewise if the low is less than the high.
    if (f_dim_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
      
    // Add the clip from this dimension to the previous results 
    f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
    f_high = (f_dim_high < f_high) ? f_dim_high : f_high;
    
    if (f_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
  }
  return make_float2(f_low, f_high);
}


__global__ 
void ray_aabb_intersection_kernel(
    float3* rays_o, 
    float3* rays_d,
    float3* aabb_center,
    float3* aabb_size,
    float2* bounds, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];
        
        bounds[taskIdx] = RayAABBIntersection(origin, direction, aabb_center[0], aabb_size[0] / 2.0f);

        taskIdx += total_thread;
    }
}

// 3d dda algorithm
struct DDASatateScene_v2{
    int3 tstep;
    float3 tMax;        // maximum t for each axis
    float3 tDelta;      // delta t for each axis
    int3 current_tile;  // could be nodes and voxels
    int3 mask;          
    float2 t;
    int3 SIDE;
    float3 TILE_SIZE;


    __device__ void init(float3 origin, float3 direction, float2 t_start, 
                        int3 _SIDE, float3 _TILE_SIZE)
    {
        
        SIDE = _SIDE; TILE_SIZE = _TILE_SIZE;
        // [TODO] may should clamp origin ?
        origin = origin + t_start.x * direction;
        current_tile = make_int3(origin / TILE_SIZE);
        current_tile = clamp(current_tile, 0, SIDE-1);

        // indicate the three directions of the marching ray
        tstep = signf(direction);

        float3 next_boundary = make_float3(current_tile + tstep) * TILE_SIZE;

        if (tstep.x < 0) next_boundary.x += TILE_SIZE.x;
        if (tstep.y < 0) next_boundary.y += TILE_SIZE.y;
        if (tstep.z < 0) next_boundary.z += TILE_SIZE.z;

        t = t_start; // init 

        tMax = fmaxf(safe_divide(next_boundary-origin, direction), 0.0f) + t.x;
        tDelta = fabs(safe_divide(TILE_SIZE, direction));

    }

    __device__ void next()
    {
        // determine the minimum t among the 3 axises
		mask.x = int((tMax.x < tMax.y) & (tMax.x <= tMax.z));
		mask.y = int((tMax.y < tMax.z) & (tMax.y <= tMax.x));
        mask.z = !(mask.x | mask.y);
		// mask.z = int((tMax.z < tMax.x) & (tMax.z <= tMax.y));
        t.y = mask.x ? tMax.x : (mask.y ? tMax.y : tMax.z);
    }

    __device__ void step()
    {
        t.x = t.y;
        tMax += make_float3(mask) * tDelta;
        current_tile += mask * tstep;
    }

    __device__ bool terminate()
    {
        return current_tile.x < 0 || current_tile.y < 0 || current_tile.z < 0 || 
               current_tile.x >= SIDE.x || current_tile.y >= SIDE.y || current_tile.z >= SIDE.z ||
               (tMax.x <= 0 && tMax.y <= 0 && tMax.z <= 0);
    }

};


inline __device__ __host__ void uniform_sample_bound_v2(
    float* zvals,
    float near, float far,
    int Nsamples)
{
    float interval = (far - near) / Nsamples;
    for (int i=0; i<Nsamples; i++)
    {
        zvals[i] = near + i * interval;
    }
}

__device__ 
void grid_pruning_single_ray(
    float3 origin,
    float3 direction,
    float3 block_corner, 
    float3 block_size,
    int* counts,
    int3 resolution)
{
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    if (bound.x == -1) return;

    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        
        atomicAdd(&counts[n], 1);

        dda.step();
    }
 
}


__global__ 
void grid_pruning_kernel(
    float3* rays_o,
    float3* rays_d,
    // minimum corner of the bbox
    float3* block_corner,
    // size of the bbox
    float3* block_size,
    int* counts,
    // resolution of the grid
    int rx, int ry, int rz,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        grid_pruning_single_ray(rays_o[taskIdx], rays_d[taskIdx],
                                block_corner[0], block_size[0], counts, make_int3(rx,ry,rz));

        taskIdx += total_thread;
    }
}




__device__ 
void sample_points_single_ray(
    float3 origin,
    float3 direction,
    int num_sample,
    float* z_vals, 
    float3 block_corner, 
    float3 block_size,
    bool* occupied_gird,
    int3 resolution)
{
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    if (bound.x == -1) return;

    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    float total_length = 0.0f;
    int count = 0;

    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = dda.t.y - dda.t.x;

            if (len > 0)
            {
                total_length += len;
                count++;
            }
        }

        dda.step();
    }

    if (count == 0) return;

    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    int left_sample = num_sample;
    int sample_count = 0;
    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        // uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = dda.t.y - dda.t.x;
            if (len > 0)
            {
                int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
                if (sample_count == count - 1) 
                {
                    num = left_sample;
                }

                uniform_sample_bound_v2(z_vals+num_sample-left_sample, dda.t.x, dda.t.y, num);

                left_sample = left_sample-num;
                sample_count++;
            }
        }

        dda.step();
    }
}


__global__ 
void sample_points_kernel(
    float3* rays_o,
    float3* rays_d,
    int num_sample, 
    float* z_vals,
    float3* block_corner,
    float3* block_size,
    bool* occupied_gird,
    int rx, int ry, int rz,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {

        sample_points_single_ray(rays_o[taskIdx], rays_d[taskIdx], num_sample, z_vals+taskIdx*num_sample,
                                block_corner[0], block_size[0], occupied_gird, make_int3(rx,ry,rz));

        taskIdx += total_thread;
    }
}



__global__ 
void update_grid_weight_kernel(
    float3* pts,
    float* weights,
    int* grid_w,
    float3* block_corner,
    float3* block_size,
    int rx, int ry, int rz,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        float3 pos = pts[taskIdx];
        int int_w = int(weights[taskIdx] * 250.0);

        int3 resolution = make_int3(rx,ry,rz);
        float3 grid_size = block_size[0] / make_float3(resolution);

        float3 origin = pos - block_corner[0];

        int3 current_tile = make_int3(origin / grid_size);
        current_tile = clamp(current_tile, 0, resolution-1);

        uint32_t n = current_tile.x * (resolution.y * resolution.z) + current_tile.y * resolution.z + current_tile.z;

        atomicMax(&grid_w[n], int_w);

        taskIdx += total_thread;
    }
}



__global__ 
void points_in_grid_kernel(
    float3* pts,
    bool* occupied_gird,
    float3* block_corner,
    float3* block_size,
    int* out_mask,
    int rx, int ry, int rz,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        float3 pos = pts[taskIdx];

        int3 resolution = make_int3(rx,ry,rz);
        float3 grid_size = block_size[0] / make_float3(resolution);

        float3 origin = pos - block_corner[0];

        int3 current_tile = make_int3(origin / grid_size);

        // out of the bounding box
        if(current_tile.x < 0 || current_tile.y < 0 || current_tile.z < 0 || current_tile.x >= resolution.x || current_tile.y >= resolution.y || current_tile.z >= resolution.z){
          out_mask[taskIdx] = 0;
        }
        else{
          uint32_t n = current_tile.x * (resolution.y * resolution.z) + current_tile.y * resolution.z + current_tile.z;
          if(occupied_gird[n]) out_mask[taskIdx] = 1;
          else out_mask[taskIdx] = 0;
        }
        
        taskIdx += total_thread;
    }
}




