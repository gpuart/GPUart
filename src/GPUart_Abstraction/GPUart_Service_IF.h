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
* File:			GPUart_Service_IF.h
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			03.04.2017								*/
/********************************************************
*   ___   ___    ___                   ___ 3      ___   *
*  |     |   |  |   |  |\  /|  |   |  |      |   |      *
*  |___  |   |  |___|  | \/ |  |   |  |___   |   |      *
*  |     |   |  |\     |    |  |   |      |  |   |      *
*  |     |___|  |  \   |    |  |___|   ___|  |   |___   *
*                                                       *
*********************************************************/


/*!	@file 	GPUart_Service_IF.h
 *
 * 	@brief 	This interface provides the service-orientated init-, call-, and query-interfaces,
 * 			respectively for each kernel.
 *
 *			All host-sided applications which must be able to trigger GPGPU kernel must include this interface.
 * 			Call a kernel's init-interface to initialize kernel related GPU data.
 * 			Call a kernel's call-interface to enqueue a new kernel instance in the scheduler and update GPU data.
 * 			Call a kernel's query-interface to query kernel completion and to get the output of the kernel.
 * 			This layer is used to achieve higher portability by abstracting the systems's heterogeneity.
 *
 * 	@author	Christoph Hartmann
 *  @date	Created on: 3 Apr 2017
 */

#ifndef GPUART_SERVICE_IF_H
#define GPUART_SERVICE_IF_H


/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/
#include "../GPUart_Common/GPUart_Common.h"



/************************************************************************************************/
/* Function declaration																			*/
/************************************************************************************************/

/*	@brief	Initialize GPU data for the Sobel1 kernel.
 *
 * 			All kernel related constant memory data can be initialized by calling this function.
 * 			Initialize global memory data here.
 * 			This function must be called before launching the first kernel instance.
 *
 * 	@param	void
 * 	@return	GPUART_SUCCESS
 */
GPUart_Retval gpuA_Sobel1_init
(
		void
);

/*	@brief	Instantiates a Sobel1 kernel instance and updates kernel related GPU data.
 *
 * 			Checks whether the scheduler is ready to enqueue a new instance of this kernel.
 * 			Updates kernel related GPU data and then instantiates kernel instance.
 *
 * 	@param[in] sint32* sob1_matrix_in_s32_swc -> Sobel1 input matrix.
 *
 * 	@return	GPUART_SUCCESS if data have been updated and kernel has been instantiated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel has already been instantiated.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (#device_global_memory_id_e)
 * 			inside this function are invalid.
 */
GPUart_Retval gpuA_Sobel1_call
(
		sint32* sob1_matrix_in_s32_swc
);

/*	@brief	Query whether Sobel1 kernel instance has completed and get kernel output data.
 *
 * 			Calls the scheduling layer (GPUart_Scheduler.h) to get the completion status of the current kernel instance and updates
 * 			the output data of this kernel.
 *
 * 	@param[out] sint32* sob1_matrix_out_s32_swc -> Sobel1 output matrix.
 *
 * 	@return	GPUART_SUCCESS if kernel instance has completed and data have been updated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel instance is still active.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (#device_global_memory_id_e)
 * 			inside this functionare invalid.
 */
GPUart_Retval gpuA_Sobel1_query
(
		sint32* sob1_matrix_out_s32_swc
);

/*	@brief	Initialize GPU data for the Sobel2 kernel.
 *
 * 			All kernel related constant memory data can be initialized by calling this function.
 * 			Initialize global memory data here.
 * 			This function must be called before launching the first kernel instance.
 *
 * 	@param	void
 * 	@return	GPUART_SUCCESS
 */
GPUart_Retval gpuA_Sobel2_init
(
		void
);


/*	@brief	Instantiates a Sobel2 kernel instance and updates kernel related GPU data.
 *
 * 			Checks whether the scheduler is ready to enqueue a new instance of this kernel.
 * 			Updates kernel related GPU data and then instantiates kernel instance.
 *
 * 	@param[in] sint32* sob2_matrix_in_s32_swc -> Sobel2 input matrix.
 *
 * 	@return	GPUART_SUCCESS if data have been updated and kernel has been instantiated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel has already been instantiated.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (#device_global_memory_id_e)
 * 			inside this function are invalid.
 */
GPUart_Retval gpuA_Sobel2_call
(
		sint32* sob2_matrix_in_s32_swc
);

/*	@brief	Query whether Sobel2 kernel instance has completed and get kernel output data.
 *
 * 			Calls the scheduling layer (GPUart_Scheduler.h) to get the completion status of the current kernel instance and updates
 * 			the output data of this kernel.
 *
 * 	@param[out] sint32* sob2_matrix_out_s32_swc -> Sobel2 output matrix.
 *
 * 	@return	GPUART_SUCCESS if kernel instance has completed and data have been updated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel instance is still active.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (#device_global_memory_id_e)
 * 			inside this functionare invalid.
 */
GPUart_Retval gpuA_Sobel2_query
(
		sint32* sob2_matrix_out_s32_swc
);

/*	@brief	Initialize GPU data for the MatrMul kernel.
 *
 * 			All kernel related constant memory data can be initialized by calling this function.
 * 			Initialize global memory data here.
 * 			This function must be called before launching the first kernel instance.
 *
 * 	@param	void
 * 	@return	GPUART_SUCCESS
 */
GPUart_Retval gpuA_MM_init
(
		void
);


GPUart_Retval gpuA_MM_call
(
		float32* mm_MatrixA_f32_swc,
		float32* mm_MatrixB_f32_swc
);

/*	@brief	Query whether MatrMul kernel instance has completed and get kernel output data.
 *
 * 			Calls the scheduling layer (GPUart_Scheduler.h) to get the completion status of the current kernel instance and updates
 * 			the output data of this kernel.
 *
 * 	@param[out] float32* mm_MatrixC_f32_swc -> Output matrix C of the MatrMul kernel (C = A x B).
 *
 * 	@return	GPUART_SUCCESS if kernel instance has completed and data have been updated successfully.
 * 	@return	GPUART_ERROR_NOT_READY if kernel instance is still active.
 * 	@return	GPUART_ERROR_INVALID_ARGUMENT if one ore more global memory IDs (#device_global_memory_id_e)
 * 			inside this function are invalid.
 */
GPUart_Retval gpuA_MM_query
(
		float32* mm_MatrixC_f32_swc
);

#endif
