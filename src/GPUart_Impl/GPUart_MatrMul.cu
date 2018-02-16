//Copyright (c) 2017-2018 Christoph A. Hartmann, Ulrich Margull and Technische Hochschule Ingolstadt (THI)
//
//Permission is hereby granted, free of charge, to any person obtaining a copy of this 
//software and associated documentation files (the "Software"), to deal in the Software
//without restriction, including without limitation the rights to use, copy, modify, 
//merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
//permit persons to whom the Software is furnished to do so, subject to the following 
//conditions:
//
//The above copyright notice and this permission notice shall be included in all copies 
//or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
//INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
//PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
//HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
//OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
//SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

/*
* File:			GPUart_MatrMul.cu
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			13.10.2017								*/
/********************************************************
*   ___   ___    ___                   ___ 3      ___   *
*  |     |   |  |   |  |\  /|  |   |  |      |   |      *
*  |___  |   |  |___|  | \/ |  |   |  |___   |   |      *
*  |     |   |  |\     |    |  |   |      |  |   |      *
*  |     |___|  |  \   |    |  |___|   ___|  |   |___   *
*                                                       *
*********************************************************/

/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/
#include "GPUart_MatrMul.cuh"

#include "GPUart_Barrier.cuh"
#include "GPUart_Impl.cuh"


/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/
#define C_MM_SM_BARRIER_TIMER 	12


/************************************************************************************************/
/* MatrMul Kernel																					*/
/************************************************************************************************/
__global__ void MatrMul_Kernel(

		//Functional Data
		float32 * __restrict__ A,
		float32 * __restrict__ B,
		float32 * __restrict__ C,

		//Preemption Buffer
		uint32 * __restrict__ block_Y_buffer,
		uint32 * __restrict__ block_X_buffer,
		uint32 * __restrict__ m_buffer,

		//Preemption Managment
		volatile sint32 * __restrict__ preemption_flag_g,
		volatile sint32 * __restrict__ preemption_sm_g,

		//Synchronization variables
		uint32 * __restrict__ sync_flags_in_u32_g,
		uint32 * __restrict__ sync_flags_out_u32_g,

		//Kernel running status
		volatile uint32 * __restrict__  kernelRunningStatus_g
)
{
	uint32 thread_Y = threadIdx.y;
	uint32 thread_X = threadIdx.x;

	uint32 block_Y;
	uint32 block_X;

	uint32 bid = blockIdx.x;
	uint32 tid = threadIdx.y * blockDim.x + threadIdx.x;

	uint32 numOfBlocks = gridDim.x;

	float32 res = 0.0f;
	uint32 m = 0;

	uint32 loopCounter = 0;

	__shared__ float32 A_s[C_MM_BLOCK_SIZE][C_MM_BLOCK_SIZE];
	__shared__ float32 B_s[C_MM_BLOCK_SIZE][C_MM_BLOCK_SIZE];

	__shared__ uint32 preemption_flag_int[1];

#ifndef C_GPUI_NO_SPATIAL_MUTLITASKING
	volatile __shared__ uint8 ghostMemory_s[(30000/C_MM_NUMBER_OF_BLOCKS) - sizeof(A_s) - sizeof(B_s) - sizeof(preemption_flag_int)];
#endif



	switch(preemption_sm_g[bid])
	{
		case C_MM_SM_INIT:
			block_Y = 0;
			block_X = blockIdx.x;
			break;
		case C_MM_SM_LOOP:
			block_Y = block_Y_buffer[bid];
			block_X = block_X_buffer[bid];
			m = m_buffer[bid];
			res = C[(block_Y * C_MM_BLOCK_SIZE + thread_Y) * C_MM_MATRIX_N + (block_X * C_MM_BLOCK_SIZE + thread_X)];
			break;
		case C_MM_SM_FINISH: break;

	}



	switch(preemption_sm_g[bid])
	{
		case C_MM_SM_INIT:


			if(tid == 0)
			{
				preemption_sm_g[bid] = C_MM_SM_LOOP;
			}


		case C_MM_SM_LOOP:

			for(; block_Y < (C_MM_MATRIX_N/C_MM_BLOCK_SIZE); block_Y++)
			{
				for(block_X = block_X % (C_MM_MATRIX_N/C_MM_BLOCK_SIZE); block_X < (C_MM_MATRIX_N/C_MM_BLOCK_SIZE); block_X += numOfBlocks)
				{

					for(; m < (C_MM_MATRIX_N/C_MM_BLOCK_SIZE); m++, loopCounter++)
					{

#ifdef S_MEAS_PREEMPTIV
						//Preemption Point
						if((loopCounter % C_MM_SM_BARRIER_TIMER == 1)&&(preemption_flag_int[0] != 0))
						{
							goto DataStoragePhase;
						}
#endif

						A_s[thread_Y][thread_X] = A[(block_Y * C_MM_BLOCK_SIZE + thread_Y) * C_MM_MATRIX_N + m * C_MM_BLOCK_SIZE + thread_X];
						B_s[thread_Y][thread_X] = B[(m * C_MM_BLOCK_SIZE + thread_Y) * C_MM_MATRIX_N + (block_X * C_MM_BLOCK_SIZE + thread_X)];

						__syncthreads();

						for(int k = 0; k < C_MM_BLOCK_SIZE; k++)
						{
							res += A_s[thread_Y][k] * B_s[k][thread_X];
						}

#ifdef S_MEAS_PREEMPTIV
						//Preemption Point
						if((loopCounter % C_MM_SM_BARRIER_TIMER == 0)&&(tid == 0))
						{
							preemption_flag_int[0] = preemption_flag_g[0];
							__threadfence();
						}
#endif

						__syncthreads();

					}
					m = 0;

					C[(block_Y * C_MM_BLOCK_SIZE + thread_Y) * C_MM_MATRIX_N + (block_X * C_MM_BLOCK_SIZE + thread_X)] = res;

					res = 0.0f;
				}
			}

			if(tid == 0)
			{
				preemption_sm_g[bid] = C_MM_SM_FINISH;
			}

		case C_MM_SM_FINISH:
			break;
	}


	DataStoragePhase: //goto-Label
	switch(preemption_sm_g[bid])
	{
		case C_MM_SM_INIT: break;

		case C_MM_SM_LOOP:
			block_Y_buffer[bid] = block_Y;
			block_X_buffer[bid] = block_X;
			m_buffer[bid] = m;
			C[(block_Y * C_MM_BLOCK_SIZE + thread_Y) * C_MM_MATRIX_N + (block_X * C_MM_BLOCK_SIZE + thread_X)] = res;
				break;

		case C_MM_SM_FINISH: break;

	}

	global_synchronize_2Dim_Binary(&sync_flags_in_u32_g[0]);

	if((bid == 0)&&(tid == 0))
	{
		if(*preemption_flag_g == 0u)
		{
			*kernelRunningStatus_g = C_KERNEL_TERMINATED_SUCESSFUL;
		}
		else
		{

			*kernelRunningStatus_g = C_KERNEL_SUSPENDED;
		}

		*preemption_flag_g = 0;
	}

#ifndef C_GPUI_NO_SPATIAL_MUTLITASKING
	if(ghostMemory_s[tid] == 0)
	{
		ghostMemory_s[tid] = tid;
	}
#endif




	return;
}
