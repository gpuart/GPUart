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
* File:			GPUart_Sobel.cuh
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

#ifndef GPUART_SOBEL_CUH
#define GPUART_SOBEL_CUH

/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/
#include "../GPUart_Common/GPUart_Common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


/************************************************************************************************/
/* Sobel1 Launch Config																			*/
/************************************************************************************************/
#define C_SOB_BLOCK_SIZE			512


#define C_SOB1_LOCAL_WORK_SIZE		C_SOB_BLOCK_SIZE		//The number of of threads (worker) within a thread block (work group)
#define	C_SOB1_NUMBER_OF_BLOCKS		1						//The overall number of blocks (work groups)
#define C_SOB1_GLOBAL_WORK_SIZE		C_SOB_BLOCK_SIZE * 	C_SOB1_NUMBER_OF_BLOCKS	//This is the overall number of threads that are calculating the investigated kernel


#if  C_SOB1_NUMBER_OF_BLOCKS != (C_SOB1_GLOBAL_WORK_SIZE / C_SOB1_LOCAL_WORK_SIZE)
#error Local work size times number of blocks doesnt equal global work size
#endif

#if  (C_SOB1_GLOBAL_WORK_SIZE % 32 != 0) || (C_SOB1_LOCAL_WORK_SIZE % 32 != 0)
#error Local as well as global work size have to be a multiple of warp size (32)
#endif



#define C_SOB1_WIDTH				C_SOB_BLOCK_SIZE
#define C_SOB1_HEIGHT				256
#define C_SOB1_MATRIX_SIZE			C_SOB1_HEIGHT * C_SOB1_WIDTH




/************************************************************************************************/
/* Sobel2 Launch Config																			*/
/************************************************************************************************/
#define C_SOB2_LOCAL_WORK_SIZE		C_SOB_BLOCK_SIZE		//The number of of threads (worker) within a thread block (work group)
#define	C_SOB2_NUMBER_OF_BLOCKS		2						//The overall number of blocks (work groups)
#define C_SOB2_GLOBAL_WORK_SIZE		C_SOB_BLOCK_SIZE * C_SOB2_NUMBER_OF_BLOCKS	//This is the overall number of threads that are calculating the investigated kernel


#if  C_SOB2_NUMBER_OF_BLOCKS != (C_SOB2_GLOBAL_WORK_SIZE / C_SOB2_LOCAL_WORK_SIZE)
#error Local work size times number of blocks doesnt equal global work size
#endif

#if  (C_SOB2_GLOBAL_WORK_SIZE % 32 != 0) || (C_SOB2_LOCAL_WORK_SIZE % 32 != 0)
#error Local as well as global work size have to be a multiple of warp size (32)
#endif



#define C_SOB2_WIDTH				C_SOB_BLOCK_SIZE * 2  	//Width = 1024
#define C_SOB2_HEIGHT				512
#define C_SOB2_MATRIX_SIZE			C_SOB2_HEIGHT * C_SOB2_WIDTH




/************************************************************************************************/
/* State Machine 																				*/
/************************************************************************************************/
#define C_SOB_SM_INIT		(0)
#define C_SOB_SM_LOOP		(1)
#define C_SOB_SM_FINISH		(2)


/************************************************************************************************/
/* Sobel Kernel																					*/
/************************************************************************************************/
__global__ void Sobel_Kernel
(
		sint32 * __restrict__ matrix_in_s32_g,
		sint32 * __restrict__ matrix_out_s32_g,
		uint32 heigth_u32,
		uint32 width_u32,

		//Preemption status variables
		volatile sint32 * 	__restrict__ preemption_flag_g,
		sint32 * 	__restrict__ preemption_flag_intern_g,
		volatile sint32 *	__restrict__ preemption_sm_g,

		//Synchronization variables
		uint32 * __restrict__ sync_flags_in_u32_g,
		uint32 * __restrict__ sync_flags_out_u32_g,

		//Buffer variables
		uint32 * __restrict__ buffer_loop_counter_u32_g,

		//Kernel running status
		volatile uint32 * __restrict__  kernelRunningStatus_g
);




#endif
