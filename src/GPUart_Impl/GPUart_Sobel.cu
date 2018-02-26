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
* File:			GPUart_Sobel.cu
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			16.05.2017								*/
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
#include "GPUart_Sobel.cuh"

#include "GPUart_Barrier.cuh"
#include "GPUart_Impl.cuh"


/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/
#define C_SOB_BARRIER_TIMER 	21


/************************************************************************************************/
/* Sobel Kernel																					*/
/************************************************************************************************/

__constant__ sint32 sobelFilter_X_u32[3][3] =
{
		{ 1, 0, -1 },
		{ 2, 0, -2 },
		{ 1, 0, -1 }
};

__constant__ sint32 sobelFilter_Y_u32[3][3] =
{
		{ 1,   2,  1 },
		{ 0,   0,  0 },
		{-1,  -2, -1 }
};



__global__ void Sobel_Kernel
(
	sint32 * matrix_in_s32_g,
	sint32 * matrix_out_s32_g,
	uint32 heigth_u32,
	uint32 width_u32,


	//Preemption status variables
	volatile sint32 * 	__restrict__ preemption_flag_g,
	sint32 * 	__restrict__ preemption_flag_intern_g,
	volatile sint32 *	__restrict__ preemption_sm_g,

	//Buffer variables
	uint32 * buffer_loop_counter_u32_g,

	//Synchronization variables
	uint32 * sync_flags_in_u32_g,
	uint32 * sync_flags_out_u32_g,

	//Kernel running status
	volatile uint32 * __restrict__  kernelRunningStatus_g
)
{
	uint32 global_id = (blockIdx.x*blockDim.x) + threadIdx.x;
	uint32 global_size = gridDim.x * blockDim.x;

	uint32 rowID_u32 = 0;
	uint32 columnID_u32 = 0;
	uint32 index_u32 = global_id;
	sint32 Gx_s32 = 0;
	sint32 Gy_s32 = 0;
	sint32 matrix_temp_s32;

	uint32 loop_condition = (heigth_u32 * width_u32 + global_size - 1)/global_size;
	uint32 loop_iteration = 0;

	uint8 preempted_u8 = C_FALSE;


	__shared__ sint32 matrix_buffer[3][C_SOB_BLOCK_SIZE + 32];




	//Initialization phase
	switch(*preemption_sm_g)
	{
	case C_SOB_SM_INIT:
		 loop_iteration = 0;
		 index_u32 = global_id;
		 break;
	case C_SOB_SM_LOOP:
		 loop_iteration = buffer_loop_counter_u32_g[0];
		 index_u32 = global_id + (loop_iteration * global_size);
		 break;
	case C_SOB_SM_FINISH:
		 break;
	}

#ifdef S_MEAS_PREEMPTIV
	global_synchronize(63, &sync_flags_in_u32_g[0],&sync_flags_out_u32_g[0]);
#endif

	//Execution Phase
	switch(*preemption_sm_g)
	{
	case C_SOB_SM_INIT:
		if(global_id == 0)
		{
			*preemption_sm_g = C_SOB_SM_LOOP;
		}
	case C_SOB_SM_LOOP:

		rowID_u32 = index_u32 / width_u32;
		columnID_u32 = index_u32 % width_u32;




		for(; loop_iteration < loop_condition; loop_iteration++, index_u32 += global_size)
		{
#ifdef S_MEAS_PREEMPTIV
			//Preemption Point
			if(loop_iteration % C_SOB_BARRIER_TIMER == 0)
			{
				if(checkpointBarrier(loop_iteration, &sync_flags_in_u32_g[0],&sync_flags_out_u32_g[0],
						  preemption_flag_g, preemption_flag_intern_g) != 0)
				{

					preempted_u8 = 1;
					break;
				}
			}
#endif

			rowID_u32 = index_u32 / width_u32;
			columnID_u32 = index_u32 % width_u32;

			Gx_s32 = 0;
			Gy_s32 = 0;




			if((rowID_u32 > 0)&&(rowID_u32 < (heigth_u32 -1)))
			{
				matrix_buffer[0][threadIdx.x+1]= matrix_in_s32_g[index_u32 - width_u32];
				matrix_buffer[1][threadIdx.x+1] = matrix_in_s32_g[index_u32];
				matrix_buffer[2][threadIdx.x+1]= matrix_in_s32_g[index_u32 + width_u32];


				if((columnID_u32 > 0)&&(columnID_u32 < (width_u32 -1)))
				{
					if(threadIdx.x == 0)
					{
						matrix_buffer[0][0]= matrix_in_s32_g[(index_u32 - width_u32) -1 ];
						matrix_buffer[1][0] = matrix_in_s32_g[index_u32 -1];
						matrix_buffer[2][0]= matrix_in_s32_g[(index_u32 + width_u32) -1];
					}
					if(threadIdx.x == C_SOB_BLOCK_SIZE - 1)
					{
						matrix_buffer[0][C_SOB_BLOCK_SIZE]= matrix_in_s32_g[(index_u32 - width_u32) + 1 ];
						matrix_buffer[1][C_SOB_BLOCK_SIZE] = matrix_in_s32_g[index_u32 + 1];
						matrix_buffer[2][C_SOB_BLOCK_SIZE]= matrix_in_s32_g[(index_u32 + width_u32) + 1];
					}


					__syncthreads();

					for(sint32 i = 0; i <= 2; i++)
					{
						for(sint32 j = 0; j <= 2; j++)
						{
							matrix_temp_s32 = matrix_buffer[i][(threadIdx.x) + j];

							Gx_s32 += sobelFilter_X_u32[i][j] * matrix_temp_s32;
							Gy_s32 += sobelFilter_Y_u32[i][j] * matrix_temp_s32;
						}
					}
					matrix_out_s32_g[index_u32] = (sint32)sqrt(float64(Gx_s32 * Gx_s32) + float64(Gy_s32 * Gy_s32));
				}
			}
		}

		if(preempted_u8 == C_TRUE)
		{
			break;
		}
		else
		{
			if(global_id == 0)
			{
					*preemption_sm_g = C_SOB_SM_FINISH;
			}
		}


	case C_SOB_SM_FINISH:
			break;
	}

#ifdef S_MEAS_PREEMPTIV
	global_synchronize(963, &sync_flags_in_u32_g[0],&sync_flags_out_u32_g[0]);
#endif

	//Data storage phase
	switch(*preemption_sm_g)
	{
	case C_SOB_SM_INIT:
		 break;
	case C_SOB_SM_LOOP:
		 if(global_id == 0)
		 {
			 buffer_loop_counter_u32_g[0] = loop_iteration;
		 }
		 break;
	case C_SOB_SM_FINISH:
		 break;
	}

	if(global_id == 0)
	{
		if(*preemption_sm_g == C_SOB_SM_FINISH)
		{
			*kernelRunningStatus_g = C_KERNEL_TERMINATED_SUCESSFUL;
		}
		else
		{

			*kernelRunningStatus_g = C_KERNEL_SUSPENDED;
		}

		*preemption_flag_g = 0;
		*preemption_flag_intern_g = 0;
	}



	return;
}
