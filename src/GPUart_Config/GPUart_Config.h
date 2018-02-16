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
* File:			GPUart_Common.h
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


#ifndef GPUART_CONFIG_H
#define GPUART_CONFIG_H


/************************************************************************************************/
/* Device Memory List																			*/
/************************************************************************************************/
typedef enum {
	//Total number of global device application variables
	E_CM_TOTAL_NR_OF_CONST_MEM_VARIABLES
}device_constant_memory_id_e;

typedef enum {
	//Sobel1 kernel task
	E_GM_ID_SOB1_MATRIX_IN,
	E_GM_ID_SOB1_MATRIX_OUT,
	//Sobel2 kernel task
	E_GM_ID_SOB2_MATRIX_IN,
	E_GM_ID_SOB2_MATRIX_OUT,
	//MatrMul kernel task
	E_GM_ID_MM_MATRIX_A,
	E_GM_ID_MM_MATRIX_B,
	E_GM_ID_MM_MATRIX_C,
	//Total number of global device application variables
	E_GM_TOTAL_NR_OF_GLOB_MEM_VARIABLES
}device_global_memory_id_e;


/************************************************************************************************/
/* Kernel List																					*/
/************************************************************************************************/
typedef enum {
	E_KTID_SOBEL1,
	E_KTID_SOBEL2,
	E_KTID_MM,
	//Total number of kernel tasks
	E_KTID_NUMBER_OF_KERNEL_TASKS
}kernel_task_id_e;


#endif
