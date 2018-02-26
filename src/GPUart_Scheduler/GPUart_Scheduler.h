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

/*!	@file 	GPUart_Scheduler.h
 *
 * 	@brief 	Interface to the GPUart Scheduling layer.
 *
 *
 * 	@author	Christoph Hartmann
 *  @date	Created on: 3 Apr 2017
 */

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

/*!
 *	@brief The possible scheduling states of a kernel.
 *
 *	Every job starts with status #E_STATUS_INIT. The possible transitions are #E_STATUS_INIT ->
 *	#E_STATUS_READY or #E_STATUS_TERMINATED -> #E_STATUS_READY when a kernel gets instantiated;
 *	#E_STATUS_RUNNING -> #E_STATUS_READY when a job has been preempted; #E_STATUS_READY ->
 *	#E_STATUS_RUNNING when the scheduler schedules a job; and #E_STATUS_RUNNING -> #E_STATUS_TERMINATED
 *	when a job completes.
 */
enum gpuS_kernelTask_status_e{
	E_STATUS_READY		= 0,	/*!< Kernel is active, but not running on the GPU. */
	E_STATUS_RUNNING 	= 1,	/*!< Kernel is currently running on the GPU */
	E_STATUS_TERMINATED	= 2,	/*!< Kernel has been completed */
	E_STATUS_INIT		= 3		/*!< Initial state at system startup */
};


/************************************************************************************************/
/* Function declaration																			*/
/************************************************************************************************/

/*
 *	@brief Executes the scheduling decision.
 *
 *		If the compiler switch S_NON_PREEMPTIVE is defined, this function schedules kernel non-preemptively,
 *		otherwise it schedules them in a limited preemptive manner. First this function updates the kernel's
 *		scheduling state and deletes completed kernels form #gpuS_JobList_s, then it updates the job list
 *		#gpuS_JobList_s with all new released kernels. After that, this function iterates through #gpuS_JobList_s
 *		and executes the scheduling decision according to the scheduling policy defined in #S_SCHED_POLICY.
 *
 *
 *	@return GPUART_SUCCESS
 *
 */
GPUart_Retval gpuS_schedule(void);



/*
 *	@brief Releases a new job and pushes it to the shared job stack #gpuS_JobStackShared_s.
 *
 *	@param kernel_task_id_e task_id_e -> The ID of the kernel which this function instatiates.
 *
 *	@return GPUART_SUCCESS if kernel has been instantiated successfully.
 *	@return GPUART_ERROR_INVALID_ARGUMENT if kernel ID is invalid.
 *	@return GPUART_ERROR_NO_OPERTATION if there is already an active instance  (job) of that kernel.
 *
 */
GPUart_Retval gpuS_new_Job(kernel_task_id_e task_id_e);


/*
 *	@brief Queries whether a job has terminated.
 *
 *	@param kernel_task_id_e task_id_e -> The ID of the kernel.
 *
 *	@return GPUART_SUCCESS if kernel has been terminated
 *	@return GPUART_ERROR_NOT_READY if kernel is still active.
 *
 */
GPUart_Retval gpuS_query_terminated(kernel_task_id_e task_id_e);


/*
 *	@brief Queries whether a kernel is ready to get instantiated.
 *
 *	@param kernel_task_id_e task_id_e -> The ID of the kernel.
 *
 *	@return GPUART_SUCCESS if kernel can get instantiated.
 *	@return GPUART_ERROR_NOT_READY if there is already an active instance (job) of that kernel.
 *
 */
GPUart_Retval gpuS_query_ready_to_call(kernel_task_id_e task_id_e);


/*
 *	@brief Initializes the GPUart Scheduling layer.
 *
 *			Initializes all variables of the Scheduling layer.
 *
 *	@return GPUART_SUCCESS
 *
 */
GPUart_Retval gpuS_init(void);


/*
 *	@brief Destroys the GPUart Scheduling layer.
 *
 *			Destroys the mutex of each #job_stack_s.
 *
 *	@return GPUART_SUCCESS
 *
 */
GPUart_Retval gpuS_destroy(void);



#endif
