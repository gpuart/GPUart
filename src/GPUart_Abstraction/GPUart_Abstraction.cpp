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
* File:			GPUart_Abstraction.cpp
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			04.04.2017								*/
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
#include "GPUart_Abstraction.h"
#include "GPUart_Service_IF.h"
#include "../GPUart_Impl/GPUart_Impl_Abstr_IF.h"
#include "../GPUart_Scheduler/GPUart_Scheduler.h"




/************************************************************************************************/
/* Variable definition																			*/
/************************************************************************************************/

/************************************************************************************************/
/* Function definition																			*/
/************************************************************************************************/

/*************************************************************************************************
Function: 		gpuA_initialize()
Description:	Initialize GPuart abstraction layer. Reset service state for all kernel tasks.
*/
GPUart_Retval gpuA_init(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}


/*************************************************************************************************
Function: 		gpuA_call(kernel_task_id_e taskID_e)
Description:	Call new job for kernel with taskID_e in GPUart scheduler
*/
GPUart_Retval gpuA_call(kernel_task_id_e taskID_e)
{
	return gpuS_new_Job(taskID_e);
}




GPUart_Retval gpuA_Sobel1_init( void )
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}


GPUart_Retval gpuA_Sobel1_call( sint32* sob1_matrix_in_s32_swc )
{

	GPUart_Retval retval = GPUART_SUCCESS;

	if(gpuS_query_ready_to_call(E_KTID_SOBEL1) == GPUART_SUCCESS)
	{
		retval |= gpuI_memcpyHost2Device(sob1_matrix_in_s32_swc, E_GM_ID_SOB1_MATRIX_IN);

		retval |= gpuA_call(E_KTID_SOBEL1);
	}
	else
	{
		retval = GPUART_ERROR_NOT_READY;
	}


	return retval;
}


GPUart_Retval gpuA_Sobel1_query( sint32* sob1_matrix_out_s32_swc )
{
	GPUart_Retval retval = GPUART_SUCCESS;

	if(gpuS_query_terminated(E_KTID_SOBEL1) == GPUART_SUCCESS)
	{
		//Kernel terminated
		retval |= gpuI_memcpyDevice2Host(sob1_matrix_out_s32_swc, E_GM_ID_SOB1_MATRIX_OUT);
	}
	else
	{
		//Kernel is still running -> Output not available
		retval = GPUART_ERROR_NOT_READY;
	}

	return retval;
}


GPUart_Retval gpuA_Sobel2_init(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}


GPUart_Retval gpuA_Sobel2_call( sint32* sob2_matrix_in_s32_swc )
{

	GPUart_Retval retval = GPUART_SUCCESS;

	if(gpuS_query_ready_to_call(E_KTID_SOBEL2) == GPUART_SUCCESS)
	{

		retval |= gpuI_memcpyHost2Device(sob2_matrix_in_s32_swc, E_GM_ID_SOB2_MATRIX_IN);


		retval |= gpuA_call(E_KTID_SOBEL2);


	}
	else
	{
		retval = GPUART_ERROR_NOT_READY;
	}

	if(retval != GPUART_SUCCESS)
	{
		//printf("\nSobel2 call Failed: %d", retval);
	}

	return retval;
}


GPUart_Retval gpuA_Sobel2_query( sint32* sob2_matrix_out_s32_swc )
{
	GPUart_Retval retval = GPUART_SUCCESS;

	if(gpuS_query_terminated(E_KTID_SOBEL2) == GPUART_SUCCESS)
	{
		//Kernel terminated
		retval |= gpuI_memcpyDevice2Host(sob2_matrix_out_s32_swc, E_GM_ID_SOB2_MATRIX_OUT);
	}
	else
	{
		//Kernel is still running -> Output not available
		retval = GPUART_ERROR_NOT_READY;
	}

	return retval;
}




GPUart_Retval gpuA_MM_init(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}


GPUart_Retval gpuA_MM_call(	float32* mm_MatrixA_f32_swc,float32* mm_MatrixB_f32_swc)
{

	GPUart_Retval retval = GPUART_SUCCESS;

	if(gpuS_query_ready_to_call(E_KTID_MM) == GPUART_SUCCESS)
	{

		retval |= gpuI_memcpyHost2Device(mm_MatrixA_f32_swc, E_GM_ID_MM_MATRIX_A);
		retval |= gpuI_memcpyHost2Device(mm_MatrixB_f32_swc, E_GM_ID_MM_MATRIX_B);

		retval |= gpuA_call(E_KTID_MM);
	}
	else
	{
		retval = GPUART_ERROR_NOT_READY;
	}

	return retval;

}

GPUart_Retval gpuA_MM_query(float32* mm_MatrixC_f32_swc)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	if(gpuS_query_terminated(E_KTID_MM) == GPUART_SUCCESS)
	{
		//Kernel terminated
		retval |= gpuI_memcpyDevice2Host(mm_MatrixC_f32_swc, E_GM_ID_MM_MATRIX_C);
	}
	else
	{
		//Kernel is still running -> Output not available
		retval = GPUART_ERROR_NOT_READY;
	}

	return retval;
}

