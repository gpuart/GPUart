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
* File:			GPUart_Impl.cuh
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


#ifndef GPUART_IMPL_CUH
#define GPUART_IMPL_CUH


/************************************************************************************************/
/* Includes																						*/
/************************************************************************************************/
/* CUDA Runtime and device driver*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "../GPUart_Common/GPUart_Common.h"
#include "../GPUart_Config/GPUart_Config.h"

/************************************************************************************************/
/* Compiler Switches																			*/
/************************************************************************************************/
#define S_MEAS_PREEMPTIV


/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/

#define C_KERNEL_READY					(0u)
#define C_KERNEL_ACTIVE					(1u)
#define C_KERNEL_TERMINATED_SUCESSFUL	(2u)
#define C_KERNEL_SUSPENDED				(3u)
#define C_KERNEL_INIT					(4u)


/************************************************************************************************/
/* Error Handling																				*/
/************************************************************************************************/

#define CUDA_RESET_ERROR(value)  {value |= cudaGetLastError(); CUDA_CHECK_RETURN(value);}

#define CUDA_CHECK_RETURN(value) cudaCheckReturn(__FILE__,__LINE__, #value, value)

static void cudaCheckReturn(const char *file, unsigned line, const char *statement, sint32 err)
{
	if (err == 0)
		return;
	printf("CUDA ERROR!!!");
	std::cerr << "CUDA ERROR! "<< statement << " returned " << "(" << err << ") at " << file << ":" << line << std::endl;

	//exit(1);
}


#endif
