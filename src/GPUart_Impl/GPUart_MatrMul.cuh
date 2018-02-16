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
* File:			GPUart_MatrMul.cuh
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

#ifndef GPUART_MM_CUH
#define GPUART_MM_CUH

/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/
#include "../GPUart_Common/GPUart_Common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


/************************************************************************************************/
/* MatrMul Launch Config																	*/
/************************************************************************************************/
#define C_MM_GLOBAL_WORK_SIZE		3072	//This is the overall number of threads that are calculating the investigated kernel

#define C_MM_LOCAL_WORK_SIZE_X		16		//The number of of threads (worker) within a thread block in x-dim(work group)
#define C_MM_LOCAL_WORK_SIZE_Y		16		//The number of of threads (worker) within a thread block in y-dim(work group)
#define C_MM_LOCAL_WORK_SIZE		(C_MM_LOCAL_WORK_SIZE_X * C_MM_LOCAL_WORK_SIZE_Y)   //The number of of threads (worker) within a thread block in (work group)

#define	C_MM_NUMBER_OF_BLOCKS_X		12		//The overall number of blocks in x dim (work groups)
#define	C_MM_NUMBER_OF_BLOCKS_Y		1		//The overall number of blocks in y dim (work groups)
#define	C_MM_NUMBER_OF_BLOCKS		(C_MM_NUMBER_OF_BLOCKS_X * C_MM_NUMBER_OF_BLOCKS_Y) //The overall number of blocks (work groups)

#if  C_MM_NUMBER_OF_BLOCKS != (C_MM_GLOBAL_WORK_SIZE / C_MM_LOCAL_WORK_SIZE)
#error Local work size times number of blocks doesnt equal global work size
#endif

#if  (C_MM_GLOBAL_WORK_SIZE % 32 != 0) || (C_MM_LOCAL_WORK_SIZE % 32 != 0)
#error Local as well as global work size have to be a multiple of warp size (32)
#endif

#define C_MM_BLOCK_SIZE				16
#define C_MM_MATRIX_N				768
#define C_MM_MATRIX_TOTAL_SIZE		C_MM_MATRIX_N * C_MM_MATRIX_N


/************************************************************************************************/
/* State Machine 																				*/
/************************************************************************************************/
#define C_MM_SM_INIT		(0)
#define C_MM_SM_LOOP		(1)
#define C_MM_SM_FINISH		(2)


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
);


#endif
