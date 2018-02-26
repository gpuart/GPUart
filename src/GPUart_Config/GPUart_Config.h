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
* File:			GPUart_Config.h
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

/*!	@file 	GPUart_Config.h
 *
 * 	@brief 	Configuration of IDs for constant memory data, global memory data and kernels.
 *
 *			The IDs for GPU related data and kernels are configured in this file. The layers of
 *			GPUart communicate by using this IDs in order to hide any implementation details.
 *
 * 	@author	Christoph Hartmann
 *  @date	Created on: 3 Apr 2017
 */

#ifndef GPUART_CONFIG_H
#define GPUART_CONFIG_H


/************************************************************************************************/
/* Device Memory List																			*/
/************************************************************************************************/

/*! @typedef device_constant_memory_id_e
 *  @brief	Defines the IDs for all constant memory data elements
 *
 *  		An unique ID must be set for each constant memory element, which must be
 *  		accessed by any host-sided application.
 *  		#E_CM_TOTAL_NR_OF_CONST_MEM_VARIABLES represents the total number of constant memory
 *  		entities, accessible via the Abstraction layer (GPUart_Service_IF.h).
 */
typedef enum {
	//Total number of global device application variables
	E_CM_TOTAL_NR_OF_CONST_MEM_VARIABLES /*!< Total number of constant memory elements */
}device_constant_memory_id_e;


/*! @typedef device_global_memory_id_e
 *  @brief	Defines the IDs for all global memory data elements
 *
 *  		An unique ID must be set for each global memory element, which must be
 *  		accessed by any host-sided application.
 *  		#E_GM_TOTAL_NR_OF_GLOB_MEM_VARIABLES represents the total number of global memory
 *  		entities, accessible via the Abstraction layer.
 */
typedef enum {
	//Sobel1 kernel task
	E_GM_ID_SOB1_MATRIX_IN,				/*!< Global Memory ID: Sobel1 input matrix*/
	E_GM_ID_SOB1_MATRIX_OUT,			/*!< Global Memory ID: Sobel1 output matrix*/
	//Sobel2 kernel task
	E_GM_ID_SOB2_MATRIX_IN,				/*!< Global Memory ID: Sobel2 input matrix*/
	E_GM_ID_SOB2_MATRIX_OUT,			/*!< Global Memory ID: Sobel2 output matrix*/
	//MatrMul kernel task
	E_GM_ID_MM_MATRIX_A,				/*!< Global Memory ID: MatrMul input matrix A*/
	E_GM_ID_MM_MATRIX_B,				/*!< Global Memory ID: MatrMul input matrix B*/
	E_GM_ID_MM_MATRIX_C,				/*!< Global Memory ID: MatrMul output matrix C*/
	//Total number of global device application variables
	E_GM_TOTAL_NR_OF_GLOB_MEM_VARIABLES	/*!< Total number of global memory elements */
}device_global_memory_id_e;


/************************************************************************************************/
/* Kernel List																					*/
/************************************************************************************************/

/*! @typedef kernel_task_id_e
 *  @brief	Defines the IDs for GPGPU kernels.
 *
 *  		An unique ID must be set for each kernel.
 *  		E_KTID_NUMBER_OF_KERNEL_TASKS represents the total number of kernels.
 */
typedef enum {
	E_KTID_SOBEL1, 						/*!< Kernel ID: Sobel1 */
	E_KTID_SOBEL2,						/*!< Kernel ID: Sobel2 */
	E_KTID_MM,							/*!< Kernel ID: MatrMul */
	//Total number of kernel tasks
	E_KTID_NUMBER_OF_KERNEL_TASKS		/*!< Total number of GPGPU kernels */
}kernel_task_id_e;


#endif
