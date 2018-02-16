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
* File:			GPUart_Barrier.cuh
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			27.04.2017								*/
/********************************************************
*   ___   ___    ___                   ___ 3      ___   *
*  |     |   |  |   |  |\  /|  |   |  |      |   |      *
*  |___  |   |  |___|  | \/ |  |   |  |___   |   |      *
*  |     |   |  |\     |    |  |   |      |  |   |      *
*  |     |___|  |  \   |    |  |___|   ___|  |   |___   *
*                                                       *
*********************************************************/


#ifndef GPUART_BARRIER_CUH
#define GPUART_BARRIER_CUH

#define C_WATCHDOG_COUNTER				(200000000)		//Software Watchdog: Timeout ticks
/************************************************************************************************/
/* Compiler Switches																			*/
/************************************************************************************************/
#define S_GLOBAL_SYNC_USE_ATOMICS						//Use atomic-functions instead of volatile variables

/************************************************************************************************/
/* Device function declaration																	*/
/************************************************************************************************/
#ifdef S_GLOBAL_SYNC_USE_ATOMICS
__device__ void global_synchronize
(
		unsigned int goalValue,
		unsigned int * __restrict__ sync_flags_in_g,
		unsigned int * __restrict__ sync_flags_out_g
);
#else
__device__ void global_synchronize
(
		unsigned int goalValue,
		volatile unsigned int * __restrict__ sync_flags_in_g,
		volatile unsigned int * __restrict__ sync_flags_out_g
);
#endif

__device__ void global_synchronize_2Dim_Binary
(
		unsigned int * __restrict__ sync_flags_g
);


#ifdef S_GLOBAL_SYNC_USE_ATOMICS
__device__ int checkpointBarrier
(
		unsigned int goalValue,
		unsigned int   * __restrict__ sync_flags_in_g,
		unsigned int   * __restrict__ sync_flags_out_g,
		volatile int  * __restrict__ terminationFlagExtern_g,
		int  * __restrict__ terminationFlagIntern_g
);
#else
__device__ int checkpointBarrier
(
		unsigned int goalValue,
		volatile unsigned int   * __restrict__ sync_flags_in_g,
		volatile unsigned int   * __restrict__ sync_flags_out_g,
		volatile int  * __restrict__ terminationFlagExtern_g,
		int  * __restrict__ terminationFlagIntern_g
);
#endif




#endif
