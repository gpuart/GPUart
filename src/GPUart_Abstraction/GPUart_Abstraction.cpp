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

/*!	@file 	GPUart_Abstraction.cpp
 *
 * 	@brief 	Implementation of the GPUart Abstraction layer.
 *
 *			Implements the service-orientated init-, call-, and query-interfaces, respectively for each kernel.
 * 			Call a kernel's init-interface to initialize kernel related GPU data.
 * 			Call a kernel's call-interface to enqueue a new kernel instance in the scheduler and update GPU data.
 * 			Call a kernel's query-interface to query kernel completion and to get the output of the kernel.
 * 			This layer is used to achieve higher portability by abstracting the systems's heterogeneity.
 *
 * 	@author	Christoph Hartmann
 *  @date	Created on: 4 Apr 2017
 */

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



/*!	@brief	Initialize the context of the GPUart Abstraction layer.
 *
 * 	@param	void
 * 	@return	GPUART_SUCCESS
 */
GPUart_Retval gpuA_init(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}

/*!	@brief	Destroy the context of the GPUart Abstraction layer.
 *
 * 	@param	void
 * 	@return	GPUART_SUCCESS
 */
GPUart_Retval gpuA_destroy(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}

/*!	@brief	Calls the Scheduling layer to enque a new kernel instance
 *
 * 	@param[in]	kernel_task_id_e taskID_e -> The ID of the kernel to be enqueued in the scheduler.
 * 	@return	GPUART_SUCCESS if kernel has been enqueued successfully.
 * 	@return GPUART_ERROR_INVALID_ARGUMENT if kernel ID is invalid.
 * 	@return GPUART_ERROR_NO_OPERTATION if kernel instance has already been enqueued.
 *
 */
/*************************************************************************************************
Function: 		gpuA_call(kernel_task_id_e taskID_e)
Description:	Call new job for kernel with taskID_e in GPUart scheduler
*/
GPUart_Retval gpuA_call(kernel_task_id_e taskID_e)
{
	return gpuS_new_Job(taskID_e);
}



/*!	@brief	Initialize GPU data for the Sobel1 kernel.
 *
 * 			All kernel related constant memory data can be initialized by calling this function.
 * 			Initialize global memory data here.
 * 			This function must be called before launching the first kernel instance.
 *
 * 	@param	void
 * 	@return	GPUART_SUCCESS
 */
GPUart_Retval gpuA_Sobel1_init( void )
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}

/*!	@brief	Instantiates a Sobel1 kernel instance and updates kernel related GPU data.
 *
 * 			Checks whether the scheduler is ready to enqueue a new instance of this kernel.
 * 			Updates kernel related GPU data and then instantiates kernel instance.
 *
 * 	@param[in] sint32* sob1_matrix_in_s32_swc -> Sobel1 input matrix.
 *
 * 	@return	GPUART_SUCCESS if data have been updated and kernel has been instantiated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel has already been instantiated.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (device_global_memory_id_e)
 * 			inside this function are invalid.
 */
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


/*!	@brief	Query whether Sobel1 kernel instance has completed and get kernel output data.
 *
 * 			Calls the scheduling layer to get the completion status of the current kernel instance and updates
 * 			the output data of this kernel.
 *
 * 	@param[out] sint32* sob1_matrix_out_s32_swc -> Sobel1 output matrix.
 *
 * 	@return	GPUART_SUCCESS if kernel instance has completed and data have been updated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel instance is still active.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (device_global_memory_id_e)
 * 			inside this functionare invalid.
 */
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


/*!	@brief	Initialize GPU data for the Sobel2 kernel.
 *
 * 			All kernel related constant memory data can be initialized by calling this function.
 * 			Initialize global memory data here.
 * 			This function must be called before launching the first kernel instance.
 *
 * 	@param	void
 * 	@return	GPUART_SUCCESS
 */
GPUart_Retval gpuA_Sobel2_init(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}


/*!	@brief	Instantiates a Sobel2 kernel instance and updates kernel related GPU data.
 *
 * 			Checks whether the scheduler is ready to enqueue a new instance of this kernel.
 * 			Updates kernel related GPU data and then instantiates kernel instance.
 *
 * 	@param[in] sint32* sob2_matrix_in_s32_swc -> Sobel2 input matrix.
 *
 * 	@return	GPUART_SUCCESS if data have been updated and kernel has been instantiated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel has already been instantiated.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (device_global_memory_id_e)
 * 			inside this function are invalid.
 */
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

	return retval;
}


/*!	@brief	Query whether Sobel2 kernel instance has completed and get kernel output data.
 *
 * 			Calls the scheduling layer to get the completion status of the current kernel instance and updates
 * 			the output data of this kernel.
 *
 * 	@param[out] sint32* sob2_matrix_out_s32_swc -> Sobel2 output matrix.
 *
 * 	@return	GPUART_SUCCESS if kernel instance has completed and data have been updated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel instance is still active.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (device_global_memory_id_e)
 * 			inside this functionare invalid.
 */
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



/*!	@brief	Initialize GPU data for the MatrMul kernel.
 *
 * 			All kernel related constant memory data can be initialized by calling this function.
 * 			Initialize global memory data here.
 * 			This function must be called before launching the first kernel instance.
 *
 * 	@param	void
 * 	@return	GPUART_SUCCESS
 */
GPUart_Retval gpuA_MM_init(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	return retval;
}


/*!	@brief	Instantiates a MatrMul kernel instance and updates kernel related GPU data.
 *
 * 			Checks whether the scheduler is ready to enqueue a new instance of this kernel.
 * 			Updates kernel related GPU data and then instantiates kernel instance.
 *
 * 	@param[in] float32* mm_MatrixA_f32_swc -> Input matrix A for MutrMul kernel (C = A x B).
 * 	@param[in] float32* mm_MatrixB_f32_swc -> Input matrix B for MutrMul kernel (C = A x B).
 *
 * 	@return	GPUART_SUCCESS if data have been updated and kernel has been instantiated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel has already been instantiated.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (device_global_memory_id_e)
 * 			inside this function are invalid.
 */
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


/*!	@brief	Query whether MatrMul kernel instance has completed and get kernel output data.
 *
 * 			Calls the scheduling layer to get the completion status of the current kernel instance and updates
 * 			the output data of this kernel.
 *
 * 	@param[out] float32* mm_MatrixC_f32_swc -> Output matrix C of the MatrMul kernel (C = A x B).
 *
 * 	@return	GPUART_SUCCESS if kernel instance has completed and data have been updated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel instance is still active.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (device_global_memory_id_e)
 * 			inside this functionare invalid.
 */
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

