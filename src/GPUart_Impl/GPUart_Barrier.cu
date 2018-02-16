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

/************************************************************************************************/
/* Includes																						*/
/************************************************************************************************/
#include "GPUart_Barrier.cuh"
#include "../GPUart_Common/GPUart_Common.h"

/************************************************************************************************/
/* Device function definition																	*/
/************************************************************************************************/

/************************************************************************************************
* This is an auxiliary function for global synchronization of all thread blocks
*
* goalValue (uint):
* 		unique synchronization ID. Synchronization flags have to be set to this value
*
* sync_flags_in_g (uint *):
* 		This is an auxiliary array that consists of flags used to synchronize.
* 		If the decentralized barrier is activated all flags are used.
* 		Incoming threadblocks write the value goalValue into their corresponding flag
*
* sync_flags_out_g (uint *):
* 		This is an auxiliary array that consists of flags used to synchronize.
* 		If the decentralized barrier is activated all flags are used.
* 		Master block writes goalValue into all fields of this array and thus, signalizes that
* 		all threadblocks are synchronized
************************************************************************************************/
#ifdef S_GLOBAL_SYNC_USE_ATOMICS

__device__ void global_synchronize
(
		unsigned int goalValue,
		unsigned int * __restrict__ sync_flags_in_g,
		unsigned int * __restrict__ sync_flags_out_g)
{
	/************************************************************************************/
	/* Decentralized Barrier															*/
	/************************************************************************************/

	int local_idx = threadIdx.x;								//local thread id
	int block_idx = blockIdx.x;									//thread block id
	int num_of_blocks_l = gridDim.x;							//number of Blocks
	int watchdog = 0;


	if( local_idx == 0u )										//Only perform read and write instructions if local id is zero
	{															//to reduce memory contention and ensure consistency
																//Set sync flag to signalize that this block already reached the barrier
		atomicExch(&sync_flags_in_g[block_idx], goalValue);
	}


	if(block_idx == 0)
	{
		if(local_idx < num_of_blocks_l)
		{
			while( atomicOr(&sync_flags_in_g[local_idx], 0) != goalValue)	//As long as not all flags are set to goalValue wait in this loop
			{
				watchdog++;
				if(watchdog > C_WATCHDOG_COUNTER)
				{
					printf("\n###############\nTimeoutWatchdogMasterBlock: %d (XF Barrier)", goalValue);
					for(int j = 0; j < gridDim.x; j++){ printf("\nIn[%d] = %d, Out[%d] = %d", j, sync_flags_in_g[j], j, sync_flags_out_g[j]);}
					asm("trap;");
				}
			}
		}

		__syncthreads();										//Make sure that all warp of the block finishes the while loop

		if(local_idx < num_of_blocks_l)
		{
			atomicExch(&sync_flags_out_g[local_idx],goalValue);			//Now every block has reached the barrier. Set flags back to 0 to open the barrier
		}
	}

	if( local_idx == 0)											//Only perform read and write instructions if local id is zero
	{															//to reduce memory contention and ensure consistency
		while( atomicOr(&sync_flags_out_g[block_idx],0) != goalValue)
		{
			watchdog++;
			if(watchdog > C_WATCHDOG_COUNTER)
			{
				printf("\n###############\nTimeoutWatchdogSlaveBlock[%d]: %d (XF Barrier)",block_idx, goalValue);
				for(int j = 0; j < gridDim.x; j++){ printf("\nIn[%d] = %d, Out[%d] = %d", j, sync_flags_in_g[j], j, sync_flags_out_g[j]);}
				asm("trap;");
			}
		}
	}

	__syncthreads();
}

#else

__device__ void global_synchronize
(
		unsigned int goalValue,
		volatile unsigned int * __restrict__ sync_flags_in_g,
		volatile unsigned int * __restrict__ sync_flags_out_g
)
{
	/************************************************************************************/
	/* Decentralized Barrier															*/
	/************************************************************************************/

	int local_idx = threadIdx.x;								//local thread id
	int block_idx = blockIdx.x;									//thread block id
	int num_of_blocks_l = gridDim.x;
	int watchdog = 0;


	if( local_idx == 0u )										//Only perform read and write instructions if local id is zero
	{															//to reduce memory contention and ensure consistency
		sync_flags_in_g[block_idx] = goalValue;					//Set sync flag to signalize that this block already reached the barrier
	}

	if(block_idx == 0)
	{
		if(local_idx < num_of_blocks_l)
		{
			do	//As long as not all flags are set to 1 wait in this loop
			{
				watchdog++;
				if(watchdog > C_WATCHDOG_COUNTER)
				{
					printf("\n###############\nTimeoutWatchdogMasterBlock: %d (XF Barrier)", goalValue);
					for(int j = 0; j < gridDim.x; j++){ printf("\nIn[%d] = %d, Out[%d] = %d", j, sync_flags_in_g[j], j, sync_flags_out_g[j]);}
					asm("trap;");
				}
			} while( sync_flags_in_g[local_idx] != goalValue );
		}

		__syncthreads();										//Make sure that all warp of the block finishes the while loop

		if(local_idx < num_of_blocks_l)
		{
			sync_flags_out_g[local_idx] = goalValue;			//Now every block has reached the barrier. Set flags back to 0 to open the barrier
		}
	}

	if( local_idx == 0)											//Only perform read and write instructions if local id is zero
	{															//to reduce memory contention and ensure consistency
		do	//As long as not all flags are set to 1 wait in this loop
		{
			watchdog++;
			if(watchdog > C_WATCHDOG_COUNTER)
			{
				printf("\n###############\nTimeoutWatchdogSlaveBlock[%d]: %d (XF Barrier)",block_idx, goalValue);
				for(int j = 0; j < gridDim.x; j++){ printf("\nIn[%d] = %d, Out[%d] = %d", j, sync_flags_in_g[j], j, sync_flags_out_g[j]);}
				asm("trap;");
			}
		} while( sync_flags_out_g[block_idx] != goalValue );
	}

	__syncthreads();
}

#endif



__device__ void global_synchronize_2Dim_Binary
(
		unsigned int * __restrict__ sync_flags_g
)
{
	/************************************************************************************/
	/* Decentralized Barrier															*/
	/************************************************************************************/

	int local_idx = threadIdx.y * blockDim.x + threadIdx.x;		//local thread id
	int block_idx = blockIdx.y * gridDim.x + blockIdx.x;		//block id
	int num_of_blocks_l = gridDim.x * gridDim.y;				//number of Blocks
	int watchdog = 0;


	if( local_idx == 0u )										//Only perform read and write instructions if local id is zero
	{															//to reduce memory contention and ensure consistency														//Set sync flag to signalize that this block already reached the barrier
		atomicExch(&sync_flags_g[block_idx], 1u);
	}


	if(block_idx == 0)
	{
		if(local_idx < num_of_blocks_l)
		{
			while( atomicOr(&sync_flags_g[local_idx], 0) != 1u)	//As long as not all flags are set to goalValue wait in this loop
			{
				watchdog++;
				if(watchdog > C_WATCHDOG_COUNTER)
				{
					printf("\n###############\nTimeoutWatchdogMasterBlock: %d (XF Barrier 2Dim)");
					for(int j = 0; j < gridDim.x; j++){ printf("\nFlag[%d] = %d", j, sync_flags_g[j]);}
					asm("trap;");
				}
			}
		}

		__syncthreads();										//Make sure that all warp of the block finishes the while loop

		if(local_idx < num_of_blocks_l)
		{
			atomicExch(&sync_flags_g[local_idx], 0u);			//Now every block has reached the barrier. Set flags back to 0 to open the barrier
		}
	}

	if( local_idx == 0)											//Only perform read and write instructions if local id is zero
	{															//to reduce memory contention and ensure consistency
		while( atomicOr(&sync_flags_g[block_idx],0) != 0u)
		{
			watchdog++;
			if(watchdog > C_WATCHDOG_COUNTER)
			{
				printf("\n###############\nTimeoutWatchdogSlaveBlock[%d]: %d (XF Barrier 2Dim)",block_idx);
				for(int j = 0; j < gridDim.x; j++){ printf("\nFlag[%d] = %d", j, sync_flags_g[j]);}
				asm("trap;");
			}
		}
	}

	__syncthreads();
}


/************************************************************************************************
* This is an auxiliary function for global synchronization and preemption checkpointing for all
* threads.
*
* goalValue (uint):
* 		unique synchronization ID. Synchronization flags have to be set to this value
*
* sync_flags_in_g (uint *):
* 		This is an auxiliary array that consists of flags used to synchronize.
* 		If the decentralized barrier is activated all flags are used.
* 		Incoming threadblocks write the value goalValue into their corresponding flag
*
* sync_flags_out_g (uint *):
* 		This is an auxiliary array that consists of flags used to synchronize.
* 		If the decentralized barrier is activated all flags are used.
* 		Master block writes goalValue into all fields of this array and thus, signalizes that
* 		all threadblocks are synchronized

************************************************************************************************/

#ifdef S_GLOBAL_SYNC_USE_ATOMICS
__device__ int checkpointBarrier
(
		unsigned int goalValue,
		unsigned int   * __restrict__ sync_flags_in_g,
		unsigned int   * __restrict__ sync_flags_out_g,
		volatile int  * __restrict__ terminationFlagExtern_g,
		int  * __restrict__ terminationFlagIntern_g
)
{
	/************************************************************************************/
	/* Decentralized Barrier															*/
	/************************************************************************************/

	int local_idx = threadIdx.x;								//local thread id
	int block_idx = blockIdx.x;									//thread block id
	int num_of_blocks_l = gridDim.x;
	int watchdog = 0;


	if( local_idx == 0u )										//Only perform read and write instructions if local id is zero
	{															//to reduce memory contention and ensure consistency
		atomicExch(&sync_flags_in_g[block_idx], goalValue);		//Set sync flag to signalize that this block already reached the barrier
	}

	if(block_idx == 0)
	{
		if(local_idx < num_of_blocks_l)
		{
			do	//As long as not all flags are set to 1 wait in this loop
			{
				watchdog++;
				if(watchdog > C_WATCHDOG_COUNTER)
				{
					printf("\n###############\nTimeoutWatchdogMasterBlock: %d (Checkpoint Barrier)", goalValue);
					for(int j = 0; j < gridDim.x; j++){ printf("\nIn[%d] = %d, Out[%d] = %d", j, sync_flags_in_g[j], j, sync_flags_out_g[j]);}
					asm("trap;");
				}
			} while( atomicOr(&sync_flags_in_g[local_idx],0) != goalValue );
		}

		if(local_idx == 0)
		{
			if(terminationFlagExtern_g[0] != 0)
			{
				atomicExch(&terminationFlagIntern_g[0], 1);
			}
		}

		__threadfence();
		__syncthreads();								//Make sure that all warp of the block finishes the while loop

		if(local_idx < num_of_blocks_l)
		{
			atomicExch(&sync_flags_out_g[local_idx], goalValue);		//Now every block has reached the barrier. Set flags back to 0 to open the barrier
		}
	}

	if( local_idx == 0)									//Only perform read and write instructions if local id is zero
	{													//to reduce memory contention and ensure consistency
		do	//As long as not all flags are set to 1 wait in this loop
		{
			watchdog++;
			if(watchdog > C_WATCHDOG_COUNTER)
			{
				printf("\n###############\nTimeoutWatchdogSlaveBlock[%d]: %d (Checkpoint Barrier)",block_idx, goalValue);
				for(int j = 0; j < gridDim.x; j++){ printf("\nIn[%d] = %d, Out[%d] = %d", j, sync_flags_in_g[j], j, sync_flags_out_g[j]);}
				asm("trap;");
			}
		} while( atomicOr(&sync_flags_out_g[block_idx],0) != goalValue );
	}

	__syncthreads();

	return *(volatile sint32 *)terminationFlagIntern_g;
}

#else

__device__ int checkpointBarrier
(
		unsigned int goalValue,
		volatile unsigned int   * __restrict__ sync_flags_in_g,
		volatile unsigned int   * __restrict__ sync_flags_out_g,
		volatile int  * __restrict__ terminationFlagExtern_g,
		int  * __restrict__ terminationFlagIntern_g
)
{
	/************************************************************************************/
	/* Decentralized Barrier															*/
	/************************************************************************************/

	int local_idx = threadIdx.x;								//local thread id
	int block_idx = blockIdx.x;									//thread block id
	int num_of_blocks_l = gridDim.x;
	int watchdog = 0;


	if( local_idx == 0u )										//Only perform read and write instructions if local id is zero
	{															//to reduce memory contention and ensure consistency
		sync_flags_in_g[block_idx] = goalValue;							//Set sync flag to signalize that this block already reached the barrier
	}

	if(block_idx == 0)
	{
		if(local_idx < num_of_blocks_l)
		{
			do	//As long as not all flags are set to 1 wait in this loop
			{

				watchdog++;
				if(watchdog > C_WATCHDOG_COUNTER)
				{
					printf("\n###############\nTimeoutWatchdogMasterBlock: %d (Checkpoint Barrier)", goalValue);
					for(int j = 0; j < gridDim.x; j++){ printf("\nIn[%d] = %d, Out[%d] = %d", j, sync_flags_in_g[j], j, sync_flags_out_g[j]);}
					asm("trap;");
				}
			} while( sync_flags_in_g[local_idx] != goalValue );
		}


		if(local_idx == 0)
		{
			if(terminationFlagExtern_g[0] != 0)
			{
				terminationFlagIntern_g[0] = 1;
			}
		}

		__threadfence();
		__syncthreads();								//Make sure that all warp of the block finishes the while loop

		if(local_idx < num_of_blocks_l)
		{
			sync_flags_out_g[local_idx] = goalValue;		//Now every block has reached the barrier. Set flags back to 0 to open the barrier
		}
	}

	if( local_idx == 0)										//Only perform read and write instructions if local id is zero
	{														//to reduce memory contention and ensure consistency
		do	//As long as not all flags are set to 1 wait in this loop
		{

			watchdog++;
			if(watchdog > C_WATCHDOG_COUNTER)
			{
				printf("\n###############\nTimeoutWatchdogSlaveBlock[%d]: %d (Checkpoint Barrier)",block_idx, goalValue);
				for(int j = 0; j < gridDim.x; j++){ printf("\nIn[%d] = %d, Out[%d] = %d", j, sync_flags_in_g[j], j, sync_flags_out_g[j]);}
				asm("trap;");
			}
		} while( sync_flags_out_g[block_idx] != goalValue );
	}

	__syncthreads();

	return terminationFlagIntern_g[0];
}
#endif
