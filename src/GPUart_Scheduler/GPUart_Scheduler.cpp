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


/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/

#include "GPUart_Scheduler.h"
#include "../GPUart_Impl/GPUart_Impl_Sched_IF.h"

//SWC Scheduler is required to get a time representation
#include "../SW-C/Scheduler/SWC_Scheduler.h"
#include "pthread.h"


/************************************************************************************************/
/* Compiler Switches																			*/
/************************************************************************************************/


#define S_STRATEGY_EDF

#ifndef S_STRATEGY_EDF
	#define S_STRATEGY_DM
#endif



//#define S_NON_PREEMPTIVE		//If defined, GPUart will schedule kernel non-preemptive

#ifdef S_NON_PREEMPTIVE
	#define S_NO_COST_SYSTEM		//If GPUart schedules kernels preemptive, GPU resource management is required
#else
	#undef S_NO_COST_SYSTEM		//If defined, GPUart will not manage GPU resources
#endif






/************************************************************************************************/
/* Defines																						*/
/************************************************************************************************/
#define C_GPUS_RESOURCE_FACTOR	1 //How many thread blocks are allowed per SM

/************************************************************************************************/
/* Typedefs																						*/
/************************************************************************************************/
typedef struct {
	kernel_task_id_e kernel_task_ID;			//A job's associated kernelTaskID
	uint32 priority;							//The priority of a kernel job
}job_s;


typedef struct {
	job_s list[E_KTID_NUMBER_OF_KERNEL_TASKS];	//The kernel job list
	uint8 count;								//Number of inserted jobs
	uint8 length;								//Length of this job list
}job_priority_list_s;



typedef struct {
	job_s stack[E_KTID_NUMBER_OF_KERNEL_TASKS];	//The kernel job stack for shared access
	volatile uint8 count;						//Number of elements in the stack
	pthread_mutex_t mutex;						//Mutex to ensure data consistency
}job_stack_s;


/************************************************************************************************/
/* Global Variables																				*/
/************************************************************************************************/

job_priority_list_s gpuS_JobList_s;	 			//This is the priorty list which stores all ready jobs

job_stack_s gpuS_JobStackShared_s;			 	//This stack is used to by all threads to push new jobs
job_stack_s gpuS_JobStack_s;					//This stack is used by the GPUart host only.
										 	 	//The shareds stack is periodically copied into this stack.
										 	 	//The GPUart host pulls the jobs from this stack and
										 	 	//dispatches them to the priority list.

sint8 gpuS_ktRunFlags_new_s8[E_KTID_NUMBER_OF_KERNEL_TASKS] = {0};
sint8 gpuS_ktRunFlags_old_s8[E_KTID_NUMBER_OF_KERNEL_TASKS] = {0};


/* Relative Deadlines of kernel tasks in µs */
static uint32 gpuS_relDeadlines_u32[E_KTID_NUMBER_OF_KERNEL_TASKS] = {
		 10000,	 //µs  -> E_KTID_SOBEL1
		100000,	 //µs  -> E_KTID_SOBEL2
	   1000000	 //µs  -> E_KTID_MM
};



/* Representation of kernel task is active */
static enum gpuS_kernelTask_status_e  gpuS_kernelTaskStatus_e[E_KTID_NUMBER_OF_KERNEL_TASKS] = {
		E_STATUS_INIT, 	//E_KTID_SOBEL1
		E_STATUS_INIT,	//E_KTID_SOBEL2
		E_STATUS_INIT	//E_KTID_MM
};

uint32 gpuS_nrOfMultiprocessor_u32 = 0u;

#ifdef S_NON_PREEMPTIVE
uint8 gpuS_usedMultiprocessors = 0u;
#endif

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

/*************************************************************************************************
Function: 		gpuS_jobStack_push
Description:	Enlist new kernel job in scheduler job stack
				The stack is shared by all threads. Thus, job stack is declared as volatile
				to avoid caching optimizations.
				Returns GPUART_SUCCESS if job could be inserted
				Returns GPUART_NO_SUCCESS if priority list is full
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


/*************************************************************************************************
Function: 		gpuS_jobStack_copy
Description:	Make a copy of the shared job stack an save its content into the non-shared
				job stack of the GPUart host.
				Returns GPUART_SUCCESS if job could be inserted
				Returns GPUART_NO_SUCCESS if priority list is full
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

/*************************************************************************************************
Function: 		gpuS_priorityList_insert
Description:	Enlist new kernel job in scheduler job list
				Returns GPUART_SUCCESS if job could be inserted
				Returns GPUART_NO_SUCCESS if priority list is full
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



/*************************************************************************************************
Function: 		gpuS_priorityList_delete
Description:	Get job from the front of the job list
				Returns GPUART_SUCCESS if job could be taken
				Returns GPUART_ERROR_INVALID_ARGUMENT if priority list element not exists for this ID
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

static GPUart_Retval gpuS_getPriority(kernel_task_id_e task_id_e, uint32 * priority)
{
	GPUart_Retval retval = GPUART_SUCCESS;

#ifdef S_STRATEGY_EDF
	*priority = swc_Scheduler_getClock() + gpuS_relDeadlines_u32[task_id_e];
#else
	*priority = gpuS_relDeadlines_u32[task_id_e];
#endif

	return retval;
}




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



#ifdef S_NON_PREEMPTIVE
				gpuS_usedMultiprocessors -= gpuI_getJobCosts(taskID);
#endif
			}
			//Check whether kernel has been suspended
			else if(gpuI_queryKernelPreempted(taskID) == C_TRUE)
			{
				gpuS_kernelTaskStatus_e[taskID] = E_STATUS_READY;

				gpuS_ktRunFlags_old_s8[taskID] = 0u;

#ifdef S_NON_PREEMPTIVE
				gpuS_usedMultiprocessors -= gpuI_getJobCosts(taskID);
#endif
			}
		}
	}
	return retval;
}

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
		if(gpuS_usedMultiprocessors >= gpuS_nrOfMultiprocessor_u32)
		{
			break;
		}

		taskID_e = gpuS_JobList_s.list[i].kernel_task_ID;

		if(gpuS_ktRunFlags_old_s8[taskID_e] == 0 )
		{

			if((gpuS_usedMultiprocessors + gpuI_getJobCosts(taskID_e)) <= gpuS_nrOfMultiprocessor_u32)
			{
				gpuS_usedMultiprocessors += gpuI_getJobCosts(taskID_e);
				gpuS_ktRunFlags_old_s8[taskID_e] = 1;
				gpuS_kernelTaskStatus_e[taskID_e] == E_STATUS_RUNNING;
				gpuI_runJob(taskID_e);
			}
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

/*************************************************************************************************
Function: 		gpuS_new_Job
Description:	Service that enlist new kernel job in the corresponding scheduler job list.
				Returns GPUART_SUCCESS if kernel task is not active.
				Returns GPUART_ERROR_NOT_READY if kernel task is already active
				(i.e. already enlistd).
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



GPUart_Retval gpuS_query_terminated(kernel_task_id_e task_id_e)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	if(gpuS_kernelTaskStatus_e[task_id_e] != E_STATUS_TERMINATED)
	{
		retval = GPUART_ERROR_NOT_READY;
	}

	return retval;
}


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



/*************************************************************************************************
Function: 		gpuS_init()
Description:	Initializes the GPUart scheduling module, thus, the job stack, job priority lists,
				globalVariables.
*/
GPUart_Retval gpuS_init(void)
{
	GPUart_Retval retval = GPUART_SUCCESS;

	//Get number of multiprocessors available on this device
#ifdef S_NO_COST_SYSTEM
	gpuS_nrOfMultiprocessor_u32 = C_UINT32_MAX;
#else
	GPUART_CHECK_RETURN(gpuI_get_NrOfMultiprocessors(&gpuS_nrOfMultiprocessor_u32, C_GPUS_RESOURCE_FACTOR));

#endif

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


/*************************************************************************************************
Function: 		gpuS_init()
Description:	Initializes the GPUart scheduling module, thus, the job stack, job priority lists,
				globalVariables.
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
