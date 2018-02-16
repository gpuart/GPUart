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
* File:			GPUart_Scheduler.h
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

#ifndef GPUART_SCHEDULER_H
#define GPUART_SCHEDULER_H



/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/
#include "../GPUart_Common/GPUart_Common.h"
#include "../GPUart_Config/GPUart_Config.h"


/************************************************************************************************/
/* Type definitions																				*/
/************************************************************************************************/
enum gpuS_kernelTask_status_e{
	E_STATUS_READY		= 0,
	E_STATUS_RUNNING 	= 1,
	E_STATUS_TERMINATED	= 2,
	E_STATUS_INIT		= 3
};


/************************************************************************************************/
/* Function declaration																			*/
/************************************************************************************************/
GPUart_Retval gpuS_init(void);
GPUart_Retval gpuS_destroy(void);

GPUart_Retval gpuS_new_Job(kernel_task_id_e task_id_e);
GPUart_Retval gpuS_query_terminated(kernel_task_id_e task_id_e);
GPUart_Retval gpuS_query_ready_to_call(kernel_task_id_e task_id_e);

GPUart_Retval gpuS_schedule(void);

#endif
