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
* File:			GPUart_Scheduler.cpp
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

/*!	@file 	GPUart_Scheduler.cpp
 *
 * 	@brief 	Implementation of the GPUart Scheduling layer.
 *
 *
 * 	@author	Christoph Hartmann
 *  @date	Created on: 3 Apr 2017
 */

/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/

#include "GPUart_Scheduler.h"
#include "../GPUart_Impl/GPUart_Impl_Sched_IF.h"

//SWC Scheduler is required to get a time base -> todo change to a proper time representation, e.g. time.h
#include "../SW-C/Scheduler/SWC_Scheduler.h"

//Memcpy, memset
#include <string.h>

//Mutex
#include <pthread.h>


/************************************************************************************************/
/* Compiler Switches																			*/
/************************************************************************************************/


/*! @def S_POLICY_EDF
 *	@brief Scheduling policy: Gang Earliest Deadline First (Gang-EDF)
 *	@see Kato et al. Gang EDF Scheduling of Parallel Task Systems, in: RTSS10, Washington, USA, 2009
 *		   http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5368128
 *	@see https://en.wikipedia.org/wiki/Earliest_deadline_first_scheduling
 */
/*! @def S_POLICY_FTP
 *	@brief Scheduling policy: Gang Fixed Task Priority (Gang-FTP). Set deadlines in
 *		   #gpuS_relDeadlines_u32 to the period times to get Gang Rate Monotonic scheduling (Gang-RM).
 *  @see Goosens et al. Gang FTP scheduling of periodic and parallel rigid real-time tasks in Real-Time
 *  and Network Systems (RTNS 2010), Tolouse France, 2010 https://arxiv.org/abs/1006.2617
 *	@see https://en.wikipedia.org/wiki/Fixed-priority_pre-emptive_scheduling
 *	@see https://en.wikipedia.org/wiki/Rate-monotonic_scheduling
 */
/*! @def S_SCHED_POLICY
 *	@brief Set S_SCHED_POLICY to either #S_POLICY_EDF or #S_POLICY_FTP.
 */

#define S_POLICY_EDF		(1)
#define S_POLICY_FTP		(2)
#define S_SCHED_POLICY 		S_POLICY_EDF


/*! @def S_NON_PREEMPTIVE
 *	@brief If S_NON_PREEMPTIVE is defined, GPUart will schedule kernels non-preemptively.
 */
//#define S_NON_PREEMPTIVE


/************************************************************************************************/
/* Defines																						*/
/************************************************************************************************/

/*! @def C_GPUS_RESOURCE_FACTOR
 *  @brief Defines the resource factor µ, that is, how many thread blocks are
 *  	   allowed per Streaming Multiprocessor
 */
#define C_GPUS_RESOURCE_FACTOR	1

/************************************************************************************************/
/* Typedefs																						*/
/************************************************************************************************/

/*!
 *	@brief Data type for job (kernel instance), including the kernel ID and the jobs priority.
 */
typedef struct {
	kernel_task_id_e kernel_task_ID;	/*!< A job's associated kernel ID */
	uint32 priority;					/*!< The priority of a kernel job. The lower the value,
											 the higher the priority (-> 0 is the highest priority) */
}job_s;


/*!
 *	@brief Data type for the priority list (gpuS_JobList_s) which holds all active jobs
 *		   (kernel instances).
 */
typedef struct {
	job_s list[E_KTID_NUMBER_OF_KERNEL_TASKS];	/*!< The kernel job list */
	uint8 count;								/*!< Number of active jobs */
	uint8 length;								/*!< Length of this job list */
}job_priority_list_s;


/*!
 *	@brief Data type for the priority stacks (gpuS_JobStackShared_s and gpuS_JobStack_s)
 *		   which hold new jobs (kernel instances).
 */
typedef struct {
	job_s stack[E_KTID_NUMBER_OF_KERNEL_TASKS];	/*!< The kernel job stack, holding new jobs */
	volatile uint8 count;						/*!< Number of elements in the stack */
	pthread_mutex_t mutex;						/*!< Mutex to ensure data consistency by mutual exclusion */
}job_stack_s;


/************************************************************************************************/
/* Global Variables																				*/
/************************************************************************************************/

/*!
 * 	@brief The priority ordered list which manages all active jobs
 */
static job_priority_list_s gpuS_JobList_s;


/*!
 *  @brief The shared stack which holds all new jobs.
 *
 *  	   This stack can be accessed by any thread via the call-interface of a kernel in the
 *  	   Abstraction layer (GPUart_Service_IF.h). Data integrity is assured by mutual exclusion.
 */
static job_stack_s gpuS_JobStackShared_s;


/*!
 *  @brief The private stack which holds all new jobs. It is a copy of #gpuS_JobStackShared_s.
 *
 *  	   This stack is used to reduce the blocking times, resulting by the mutual exclusive access,
 *  	   by copying the content of #gpuS_JobStackShared_s into this stack. The scheduler periodically
 *  	   pulls new jobs from this stack and inserts them into gpuS_JobList_s in a priority order.
 */
static job_stack_s gpuS_JobStack_s;


/*!
 *  @brief An array which holds the current running status of each kernel.
 *
 *  	   The length of this array is equal to the number of kernels #E_KTID_NUMBER_OF_KERNEL_TASKS.
 *  	   The i'th element represents the i'th kernel, according to the enum #kernel_task_id_e.
 *  	   If an element is equal to 0, then this kernel is NOT running currently. Otherwise, the
 *  	   corresponding kernel is running. The array gets updated with each scheduling decision.
 */
static sint8 gpuS_ktRunFlags_new_s8[E_KTID_NUMBER_OF_KERNEL_TASKS] = {0};

/*!
 *  @brief An array which holds the running status of each kernel from the last scheduling decision.
 *
 *  	   See #gpuS_ktRunFlags_new_s8
 */
static sint8 gpuS_ktRunFlags_old_s8[E_KTID_NUMBER_OF_KERNEL_TASKS] = {0};


/*!
 *	@brief An array for the configuration of the kernels' relative deadlines in µs.
 *
 *		The length of this array is equal to the number of kernels #E_KTID_NUMBER_OF_KERNEL_TASKS.
 *  	The i'th element represents the i'th kernel, according to the enum #kernel_task_id_e.
 */
static uint32 gpuS_relDeadlines_u32[E_KTID_NUMBER_OF_KERNEL_TASKS] = {
		 10000,	 //µs  -> E_KTID_SOBEL1
		100000,	 //µs  -> E_KTID_SOBEL2
	    200000	 //µs  -> E_KTID_MM
};



/*!
 *	@brief An array which represents the state machine for each kernel.
 *
 *		The length of this array is equal to the number of kernels #E_KTID_NUMBER_OF_KERNEL_TASKS.
 *  	The i'th element represents the i'th kernel, according to the enum #kernel_task_id_e. A
 *  	kernel can be either INIT, READY, RUNNING, or TERMINATED.
 */
static enum gpuS_kernelTask_status_e  gpuS_kernelTaskStatus_e[E_KTID_NUMBER_OF_KERNEL_TASKS] = {
		E_STATUS_INIT, 	/*!< E_KTID_SOBEL1. Initial State of kernel Sobel1 */
		E_STATUS_INIT,	/*!< E_KTID_SOBEL2. Initial State of kernel Sobel2 */
		E_STATUS_INIT	/*!< E_KTID_MM. Initial State of kernel MatrMul */
};

/*!
 *	@brief Represents the number of Streaming Multiprocessors m on the GPU.
 */
static uint32 gpuS_nrOfMultiprocessor_u32 = 0u;


/************************************************************************************************/
/* Function declarartion																		*/
/************************************************************************************************/
static GPUart_Retval gpuS_priorityList_insert(job_priority_list_s * pJobList, job_s pJob);
static GPUart_Retval gpuS_priorityList_delete(job_priority_list_s * pJobList, uint8 pId);
static GPUart_Retval gpuS_jobStack_push(job_stack_s * pJobStack, job_s pJob);
static GPUart_Retval gpuS_jobStack_copy(job_stack_s * pJobStackDest, job_stack_s * pJobStackSrc);
static GPUart_Retval gpuS_jobStack_pull(job_stack_s * pJobStack, job_s * pJob);
static GPUart_Retval gpuS_getPriority(kernel_task_id_e task_id_e, uint32 * priority);





/************************************************************************************************/
/* Job list related functions																	*/
/************************************************************************************************/


/*!
 *	@brief Push a new kernel instance to the shared job stack.
 *
 *		This function is called, when a new kernel instance has been activated due to a call to
 *		a kernel's call-function in GPUart_Service_IF.h. The new job is pushed on the shared job
 *		stack, which is accessed in a mutual exlusive manner.
 *	@param[out] job_stack_s * pJobStack -> The shared job stack
 *	@param[in] job_s pJob -> The new job to be released
 *
 *	@return GPUART_SUCCESS if job could be inserted
 *	@return GPUART_NO_SUCCESS if priority list is full
 *
 *	@see #gpuS_JobStackShared_s
 */
static GPUart_Retval gpuS_jobStack_push( job_stack_s * pJobStack, job_s pJob)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	pthread_mutex_lock(&pJobStack->mutex);

	if(pJobStack->count >= E_KTID_NUMBER_OF_KERNEL_TASKS)
	{
		printf("Job stack is full!!");
		retval = GPUART_NO_SUCCESS;
	}
	else
	{
		pJobStack->stack[pJobStack->count] = pJob;
		pJobStack->count++;
	}

	pthread_mutex_unlock(&pJobStack->mutex);

	return retval;
};


/*!
 *	@brief Copies a job stack #job_stack_s.
 *
 *		Make a copy of the shared job stack an save its content into the non-shared
 *		job stack of the GPUart scheduling host.
 *
 *	@param[in] job_stack_s * pJobStackDest -> The #job_stack_s to be copied.
 *	@param[out] job_stack_s * pJobStackSrc -> The copy of pJobStackDest
 *
 *	@return GPUART_SUCCESS
 *
 *	@see #gpuS_JobStackShared_s
 *	@see #gpuS_JobStack_s
 */
static GPUart_Retval gpuS_jobStack_copy(job_stack_s * pJobStackDest,  job_stack_s * pJobStackSrc)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	pthread_mutex_lock(&pJobStackSrc->mutex);

	//Copy shared stack to non-shared stack
	memcpy(&pJobStackDest->stack[0], &pJobStackSrc->stack[0], E_KTID_NUMBER_OF_KERNEL_TASKS * sizeof(job_s));
	pJobStackDest->count = pJobStackSrc->count;

	//Reset shared stack
	pJobStackSrc->count = 0;

	pthread_mutex_unlock(&pJobStackSrc->mutex);

	return retval;
};


/*************************************************************************************************
Function: 		gpuS_jobStack_pull
Description:	Takes the next available job of the job stack. No mutex is required, since this
				function is only used to read the jobs from the non-shared GPUart host job stack
				Returns GPUART_SUCCESS if job could be inserted
				Returns GPUART_NO_SUCCESS if priority list is empty
*/
/*!
 *	@brief Pull a job from a job stack.
 *
 *		Befor each scheduling decision, the scheduler pulls all jobs from #gpuS_JobStack_s and insert them into
 *		#gpuS_JobList_s in priority based order.
 *
 *	@param[in] job_stack_s * pJobStack -> The #job_stack_s from which this function pulls a #job_s.
 *	@param[out] job_s * pJob -> The #job_s to be pulled.
 *
 *	@return GPUART_SUCCESS if job has been pulled successfully.
 *	@return GPUART_NO_SUCCESS if pJobStack is empty.
 *
 *	@see #gpuS_JobStack_s
 *	@see #gpuS_JobList_s
 */
static GPUart_Retval gpuS_jobStack_pull(job_stack_s * pJobStack, job_s * pJob)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	if(pJobStack->count == 0)
	{
		retval = GPUART_NO_SUCCESS;
	}
	else
	{
		pJobStack->count--;
		*pJob = pJobStack->stack[pJobStack->count];
	}

	return retval;
};


/*!
 *	@brief Insert a job into a priority based list.
 *
 *		Befor each scheduling decision, the scheduler pulls all jobs from #gpuS_JobStack_s and insert them into
 *		#gpuS_JobList_s in priority based order. The job list #gpuS_JobList_s contains all active kernels.
 *
 *	@param[out] job_priority_list_s * pJobList -> The #job_priority_list_s in which this function inserts a job.
 *	@param[in] job_s pJob -> The #job_s to be inserted.
 *
 *	@return GPUART_SUCCESS if job has been inserted successfully.
 *	@return GPUART_NO_SUCCESS if pJobList is full.
 *
 *	@see #gpuS_JobList_s
 */
static GPUart_Retval gpuS_priorityList_insert(job_priority_list_s * pJobList, job_s pJob)
{
	GPUart_Retval retval = GPUART_SUCCESS;
	uint32 priority   	 = pJob.priority;
	sint32 idx 			 = 0;


	//Check whether job list is full
	if((pJobList->count + 1) > pJobList->length)
	{
		printf("\nPriority job list is full!!");
		//Job list is full. Return failure
		retval = GPUART_NO_SUCCESS;
	}
	else
	{
		//Job list is not full. Find the position in the list the new element should be inserted
		for(idx = pJobList->count; idx > 0; idx--)
		{
			if(pJobList->list[idx-1].priority < priority)
			{
				break;
			}
			else
			{
				pJobList->list[idx] = pJobList->list[idx-1];
			}
		}

		//Position in list is determined. Now insert new element and inkrement element counter
		pJobList->list[idx] = pJob;
		pJobList->count++;
	}


	return retval;
}




/*!
 *	@brief Delete a job from a priority based list.
 *
 *		After a job (kernel instance) has been completed, it is deleted from the list #gpuS_JobList_s
 *
 *	@param[out] job_priority_list_s * pJobList -> The #job_priority_list_s from which this function deletes a job.
 *	@param[in] uint8 pId -> The index of the list element wich is deleted by this function.
 *
 *	@return GPUART_SUCCESS if job has been deleted successfully.
 *	@return GPUART_ERROR_INVALID_ARGUMENT if pID is out of bounds.
 *
 *	@see #gpuS_JobList_s
 */
static GPUart_Retval gpuS_priorityList_delete(job_priority_list_s * pJobList, uint8 pId)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	//Check whether job list is full
	if(pId >= pJobList->count)
	{
		printf("\nId for priority list is out of bounds!!");
		//Job list is full. Return failure
		retval = GPUART_ERROR_INVALID_ARGUMENT;
	}
	else
	{
		//decrement number of elements in the list
		pJobList->count--;

		//Shift all subsequent elements one position towards front
		memmove(&(pJobList->list[pId]), &(pJobList->list[pId+1]), (pJobList->count - pId) * sizeof(job_s));
	}

	return retval;
}



/************************************************************************************************/
/* Scheduling functions																			*/
/************************************************************************************************/

/*!
 *	@brief Get the priority of a job (kernel instance).
 *
 *		Returns the priority of a job. The smaller the value, the higher the priority.
 *
 *	@param[in] kernel_task_id_e task_id_e, -> The kernel for which the priority should be returned.
 *	@param[out] uint8 pId -> The priority of the kernel instance.
 *
 *	@return GPUART_SUCCESS
 *
 */
static GPUart_Retval gpuS_getPriority(kernel_task_id_e task_id_e, uint32 * priority)
{
	GPUart_Retval retval = GPUART_SUCCESS;

#if S_SCHED_POLICY == S_POLICY_EDF
	*priority = swc_Scheduler_getClock() + gpuS_relDeadlines_u32[task_id_e];
#else
	*priority = gpuS_relDeadlines_u32[task_id_e];
#endif

	return retval;
}



/*!
 *	@brief Updates a priority based job list with all new released jobs.
 *
 *		Befor each scheduling decision, the scheduler pulls all jobs from #gpuS_JobStack_s and insert them into
 *		#gpuS_JobList_s in priority based order. The job list #gpuS_JobList_s contains all active kernels.
 *
 *
 *	@return GPUART_SUCCESS if job list could be updated successfully.
 *	@return GPUART_NO_SUCCESS if job list is full.
 *
 */
static GPUart_Retval gpuS_updateListWithNewJobs(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;
	job_s job_temp_s;


	//Copy shared job stack into own job stack
	gpuS_jobStack_copy(&gpuS_JobStack_s, &gpuS_JobStackShared_s);

	//Dispatch all new jobs to the priority list
	while(gpuS_jobStack_pull(&gpuS_JobStack_s, &job_temp_s) == GPUART_SUCCESS)
	{
		retval |= gpuS_priorityList_insert(&gpuS_JobList_s, job_temp_s);
	}

	return retval;
}


/*!
 *	@brief Updates a the scheduling state (#gpuS_kernelTaskStatus_e) of each kernel.
 *
 *		Befor each scheduling decision the scheduler polls each active job in #gpuS_JobList_s if it has been terminated or preempted.
 *		Terminated jobs are deleted in #gpuS_JobList_s by calling #gpuS_priorityList_delete(). The state
 *		of the corresponding kernel instance is set to #E_STATUS_TERMINATED. If a kernel has been preempted, its
 *		state set to #E_STATUS_READY.
 *
 *
 *	@return GPUART_SUCCESS
 *
 */
static GPUart_Retval gpuS_updateRunningStatus(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	kernel_task_id_e taskID;

	for(int j = 0; j < gpuS_JobList_s.count; j++)
	{
		taskID = gpuS_JobList_s.list[j].kernel_task_ID;


		//if(gpuS_ktRunFlags_old_s8[taskID] == 1u)
		{
			//Check whether kernel already have terminated successfully
			if(gpuI_queryKernelTerminatedSuccessful(taskID) == C_TRUE)
			{
				//Update kernel task status for upper sw-components
				gpuS_kernelTaskStatus_e[taskID] = E_STATUS_TERMINATED;

				gpuS_ktRunFlags_old_s8[taskID] = 0u;

				gpuS_priorityList_delete(&gpuS_JobList_s,  j);
			}
			//Check whether kernel has been suspended
			else if(gpuI_queryKernelPreempted(taskID) == C_TRUE)
			{
				gpuS_kernelTaskStatus_e[taskID] = E_STATUS_READY;

				gpuS_ktRunFlags_old_s8[taskID] = 0u;


			}
		}
	}
	return retval;
}


/*!
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
#ifdef S_NON_PREEMPTIVE
GPUart_Retval gpuS_schedule(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;
	kernel_task_id_e taskID_e;

	//Update the status of all running jobs
	GPUART_CHECK_RETURN( gpuS_updateRunningStatus() );

	//Update the ready-list with all new jobs
	GPUART_CHECK_RETURN( gpuS_updateListWithNewJobs() );

	for(int i = 0; i < gpuS_JobList_s.count; i++)
	{
		taskID_e = gpuS_JobList_s.list[i].kernel_task_ID;

		if(gpuS_ktRunFlags_old_s8[taskID_e] == 0 )
		{
				gpuS_ktRunFlags_old_s8[taskID_e] = 1;
				gpuS_kernelTaskStatus_e[taskID_e] == E_STATUS_RUNNING;
				gpuI_runJob(taskID_e);

		}
	}

	return retval;
}
#else
GPUart_Retval gpuS_schedule(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;
	sint32 taskID_s32;
	kernel_task_id_e taskID_e;
	sint8  edge_s8;
	uint32 costs_u32 = 0u;

	//Update the status of all running jobs
	GPUART_CHECK_RETURN( gpuS_updateRunningStatus() );

	//Update the ready-list with all new jobs
	GPUART_CHECK_RETURN( gpuS_updateListWithNewJobs() );

	//Define the new set of running tasks
	for(int i = 0; ((i < gpuS_JobList_s.count) && (costs_u32 < gpuS_nrOfMultiprocessor_u32)); i++)
	{
		taskID_e = gpuS_JobList_s.list[i].kernel_task_ID;


		if((costs_u32 + gpuI_getJobCosts(taskID_e)) <= gpuS_nrOfMultiprocessor_u32)
		{
			costs_u32 = costs_u32 + gpuI_getJobCosts(taskID_e);
			gpuS_ktRunFlags_new_s8[taskID_e] = 1;
		}
		else
		{
			//Do nothing
		}
	}


	//Check which changes in the 'run'-set occured and preempt or start kernels respectively.
	for(taskID_s32 = 0; taskID_s32 < E_KTID_NUMBER_OF_KERNEL_TASKS; taskID_s32++)
	{

		//Rising edge -> The kernel became running since the last schedule
		if((gpuS_ktRunFlags_new_s8[taskID_s32] == 1)&&(gpuS_ktRunFlags_old_s8[taskID_s32] == 0))
		{

				gpuI_runJob((kernel_task_id_e)taskID_s32);
				gpuS_kernelTaskStatus_e[taskID_s32] == E_STATUS_RUNNING;
		}

		else if((gpuS_ktRunFlags_new_s8[taskID_s32] == 0)&&(gpuS_ktRunFlags_old_s8[taskID_s32] == 1))
		{
			gpuI_preemptJob((kernel_task_id_e)taskID_s32);
			gpuS_kernelTaskStatus_e[taskID_s32] == E_STATUS_READY;
		}
		else if((gpuS_ktRunFlags_new_s8[taskID_s32] == 1)&&(gpuS_ktRunFlags_old_s8[taskID_s32] == 1))
		{
			gpuI_runJob((kernel_task_id_e)taskID_s32);
			//Nothing changed since the last schedule. Kernel is still running or still not running
		}
		else if((gpuS_ktRunFlags_new_s8[taskID_s32] == 0)&&(gpuS_ktRunFlags_old_s8[taskID_s32] == 0))
		{
			gpuI_preemptJob((kernel_task_id_e)taskID_s32);
		}
	}

	//Save current flags in "old" array and reset "new" array
	for(int i = 0; i < E_KTID_NUMBER_OF_KERNEL_TASKS; i++)
	{
		gpuS_ktRunFlags_old_s8[i] = gpuS_ktRunFlags_new_s8[i];
		gpuS_ktRunFlags_new_s8[i] = 0;
	}

	return retval;
}
#endif

/************************************************************************************************/
/* Definition of Scheduling Services															*/
/************************************************************************************************/

/*!
 *	@brief Releases a new job and pushes it to the shared job stack #gpuS_JobStackShared_s.
 *
 *	@param kernel_task_id_e task_id_e -> The ID of the kernel which this function instatiates.
 *
 *	@return GPUART_SUCCESS if kernel has been instantiated successfully.
 *	@return GPUART_ERROR_INVALID_ARGUMENT if kernel ID is invalid.
 *	@return GPUART_ERROR_NO_OPERTATION if there is already an active instance  (job) of that kernel.
 *
 */
GPUart_Retval gpuS_new_Job(kernel_task_id_e task_id_e)
{
	GPUart_Retval retval = GPUART_SUCCESS;
	job_s newJob_s;

	if(task_id_e >= E_KTID_NUMBER_OF_KERNEL_TASKS)
	{
		printf("\nInvalide kernel ID for new job");
		retval = GPUART_ERROR_INVALID_ARGUMENT;
	}
	else if((gpuS_kernelTaskStatus_e[task_id_e] != E_STATUS_TERMINATED)&&
			(gpuS_kernelTaskStatus_e[task_id_e] != E_STATUS_INIT))
	{
		printf("\nTask kernel is already new job");
		retval = GPUART_ERROR_NO_OPERTATION;
	}
	else
	{
		gpuS_kernelTaskStatus_e[task_id_e] = E_STATUS_READY;
		gpuI_SetKernelStatusReady(task_id_e);

		//Set kernel task ID of new job to identify context
		newJob_s.kernel_task_ID = task_id_e;

		//Set priority of new job
		gpuS_getPriority(task_id_e, &newJob_s.priority);

		//Enlist new job in determined job list
		gpuS_jobStack_push(&gpuS_JobStackShared_s, newJob_s);
	}

	return retval;
}


/*!
 *	@brief Queries whether a job has terminated.
 *
 *	@param kernel_task_id_e task_id_e -> The ID of the kernel.
 *
 *	@return GPUART_SUCCESS if kernel has been terminated
 *	@return GPUART_ERROR_NOT_READY if kernel is still active.
 *
 */
GPUart_Retval gpuS_query_terminated(kernel_task_id_e task_id_e)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	if(gpuS_kernelTaskStatus_e[task_id_e] != E_STATUS_TERMINATED)
	{
		retval = GPUART_ERROR_NOT_READY;
	}

	return retval;
}

/*!
 *	@brief Queries whether a kernel is ready to get instantiated.
 *
 *	@param kernel_task_id_e task_id_e -> The ID of the kernel.
 *
 *	@return GPUART_SUCCESS if kernel can get instantiated.
 *	@return GPUART_ERROR_NOT_READY if there is already an active instance (job) of that kernel.
 *
 */
GPUart_Retval gpuS_query_ready_to_call(kernel_task_id_e task_id_e)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	if((gpuS_kernelTaskStatus_e[task_id_e] == E_STATUS_TERMINATED)||
			(gpuS_kernelTaskStatus_e[task_id_e] == E_STATUS_INIT))
	{
		//Kernel task is ready to become 'ready'
	}
	else
	{
		retval = GPUART_ERROR_NOT_READY;
	}

	return retval;
}



/*!
 *	@brief Initializes the GPUart Scheduling layer.
 *
 *			Initializes all variables of the Scheduling layer.
 *
 *	@return GPUART_SUCCESS
 *
 */
GPUart_Retval gpuS_init(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	//Get number of multiprocessors available on this device
	GPUART_CHECK_RETURN(gpuI_get_NrOfMultiprocessors(&gpuS_nrOfMultiprocessor_u32, C_GPUS_RESOURCE_FACTOR));


	//Init the shared job stack
	gpuS_JobStackShared_s.count = 0u;
	GPUART_CHECK_RETURN(pthread_mutex_init(&gpuS_JobStackShared_s.mutex, NULL));

	//Init the GPUart host job stack
	gpuS_JobStack_s.count = 0u;
	GPUART_CHECK_RETURN(pthread_mutex_init(&gpuS_JobStack_s.mutex, NULL));

	//Init the GPUart host job list
	gpuS_JobList_s.count  = 0u;
	gpuS_JobList_s.length = E_KTID_NUMBER_OF_KERNEL_TASKS;

	for(int i = 0; i<E_KTID_NUMBER_OF_KERNEL_TASKS;i++)
	{
		gpuS_ktRunFlags_new_s8[i] = 0;
		gpuS_ktRunFlags_old_s8[i] = 0;
		gpuS_kernelTaskStatus_e[i] = E_STATUS_INIT;
	}


	return retval;
}


/*!
 *	@brief Destroys the GPUart Scheduling layer.
 *
 *			Destroys the mutex of each #job_stack_s.
 *
 *	@return GPUART_SUCCESS
 *
 */
GPUart_Retval gpuS_destroy(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	//Destroy stack mutex of shared job stack
	GPUART_CHECK_RETURN(pthread_mutex_destroy(&gpuS_JobStackShared_s.mutex));

	//Destroy stack mutex of GPUart host job stack
	GPUART_CHECK_RETURN(pthread_mutex_destroy(&gpuS_JobStack_s.mutex));


	return retval;
}
