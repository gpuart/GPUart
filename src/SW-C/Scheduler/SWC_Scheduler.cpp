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
* File:			GPUart_Impl.cpp
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			03.04.2017								*/
/********************************************************
*   ___   ___    ___                   ___        ___   *
*  |     |   |  |   |  |\  /|  |   |  |      |   |      *
*  |___  |   |  |___|  | \/ |  |   |  |___   |   |      *
*  |     |   |  |\     |    |  |   |      |  |   |      *
*  |     |___|  |  \   |    |  |___|   ___|  |   |___   *
*                                                       *
*********************************************************/


/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/

#include "../Sobel1/SWC_Sobel1.h"
#include "../Sobel2/SWC_Sobel2.h"
#include "../MatrMul/SWC_MM.h"
#include "SWC_Scheduler.h"


#include "../../GPUart/GPUart.h"

#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


/************************************************************************************************/
/* Compiler Switches																			*/
/************************************************************************************************/
//#define S_PRINT_TIMER_PERIOD_TIME

/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/
//Number of thread created for task handling
#define C_SWC_SCH_NUMBER_OF_THREADS			(4)

//Define how often the timer should expire until the system terminates
//The operation time is (x) * C_SWC_SCH_TIMER_RESOLUTION
#define C_SWC_SCH_SYSTEM_OPERATION_TIME 	(100000)

//The resolution of the real-time timer in nanoseconds (x * 1us)
//IF THIS VALUE CHANGED YOU HAVE TO ADAPT THE PERIODIC TIMES in swc_sch_task_periods
#define C_SWC_SCH_TIMER_RESOLUTION			(100 * 1000/*ns -> 0.1 ms resolution*/)

//The number of different period times occuring in the task schedule
#define C_SWC_SCH_NR_DIFFERENT_PERIODS		(10)


//ID of particular perodic time
#define C_SWC_SCH_PERIOD_1MS				(0)
#define C_SWC_SCH_PERIOD_2MS				(1)
#define C_SWC_SCH_PERIOD_5MS				(2)
#define C_SWC_SCH_PERIOD_10MS				(3)
#define C_SWC_SCH_PERIOD_20MS				(4)
#define C_SWC_SCH_PERIOD_50MS				(5)
#define C_SWC_SCH_PERIOD_100MS				(6)
#define C_SWC_SCH_PERIOD_200MS				(7)
#define C_SWC_SCH_PERIOD_500MS				(8)
#define C_SWC_SCH_PERIOD_1000MS				(9)

/************************************************************************************************/
/* Global Variable declaration																	*/
/************************************************************************************************/


/* Possible periodic times of this system -> Resulting periodic time is x * C_SWC_SCH_TIMER_RESOLUTION */
const static unsigned int swc_sch_task_periods[C_SWC_SCH_NR_DIFFERENT_PERIODS] =
{
		10u,	//0 -> 1ms
		20u,	//1 -> 2ms
		50u,	//2 -> 5ms
		100u,	//3 -> 10ms
		200u,	//4 -> 20ms
		500u,	//5 -> 50ms
		1000u,	//6 -> 100ms
		2000u,  //7 -> 200ms
		5000u,	//8 -> 500ms
		10000u	//9 -> 1000ms
};


/* This array holds the offset for task activation for all SW-C Tasks -> Resulting offset time is X * C_SWC_SCH_TIMER_RESOLUTION*/
const static unsigned int swc_sch_task_activ_offsets[C_SWC_SCH_NUMBER_OF_THREADS - 1][C_SWC_SCH_NR_DIFFERENT_PERIODS] =
{
/*			1ms,	2ms,	5ms,	10ms,	20ms,	50ms,	100ms,	200ms, 500ms,	1000ms	*/
/*Core0*/ {	0,		0,		0,		20,		0,		0,		0,		0,		0,		0	},
/*Core1*/ {	0,		0,		0,		0,		0,		0,		0,		0,		0,		750	},
/*Core2*/ {	0,		0,		0,		0,		0,		0,		0,		0,		0,		0	}
};



/* Stores information wheter a particular task is used or not */
static unsigned char swc_sch_task_used[C_SWC_SCH_NUMBER_OF_THREADS - 1][C_SWC_SCH_NR_DIFFERENT_PERIODS] =
{
/*			1ms,	2ms,	5ms,	10ms,	20ms,	50ms,	100ms,	200ms,	500ms,	1000ms	*/
/*Core0*/ {	0,		0,		0,		1,		0,		0,		1,		0,		0,		0	},
/*Core1*/ {	0,		0,		0,		0,		0,		0,		0,		1,		0,		1	},
/*Core2*/ {	0,		0,		0,		0,		0,		0,		0,		0,		0,		0	}
};





static volatile unsigned char swc_sch_task_event[C_SWC_SCH_NUMBER_OF_THREADS - 1][C_SWC_SCH_NR_DIFFERENT_PERIODS];

static volatile unsigned char swc_sch_terminate_threads = 0u;

static unsigned int swc_Scheduler_clock = 0u;

/************************************************************************************************/
/* Function declaration																			*/
/************************************************************************************************/
static void*	swc_Scheduler_MPMD(void *args);
static void		swc_Scheduler_Timer(void);
static void		swc_Scheduler_Thread0(void);
static void		swc_Scheduler_Thread1(void);
static void		swc_Scheduler_Thread2(void);

/************************************************************************************************/
/* Function definition																			*/
/************************************************************************************************/

/* Returns the current clock. The resolution is 1µs */
unsigned int swc_Scheduler_getClock(void)
{
	return swc_Scheduler_clock;
}


/*************************************************************************************************
Function: 		swc_init
Description:	Calls the initialization function for all SW-Cs
*/
void swc_init(void)
{
	//Initialize other SW-Cs
	swc_Sobel1_init();
	swc_Sobel2_init();
	swc_MatrMul_init();
}


/*************************************************************************************************
Function: 		swc_Scheduler_init
Description:	Initializing the SW-C scheduler
*/
void swc_Scheduler_init(void)
{
	//Initialize this module
	memset((void*)swc_sch_task_event, 0u, sizeof(swc_sch_task_event));
	swc_sch_terminate_threads = 0u;
	swc_Scheduler_clock = 0u;


}



/*************************************************************************************************
Function: 		swc_Scheduler_Timer
Description:	This thread is used for real-time timing only. It triggers execution events
				for the other tasks by setting their corresponding event flags if the periodic time
				expired.
*/
static void swc_Scheduler_Timer(void)
{
	struct timespec tstart_l = {0,0}, tende_l = {0,0};
	float measresult;
	unsigned long long expiredTime = 0;

	printf("\nstartTimer");
	fflush(stdout);


	for(int operating_time = 0; operating_time < C_SWC_SCH_SYSTEM_OPERATION_TIME; operating_time++)
	{
		clock_gettime(CLOCK_REALTIME, &tstart_l );

		usleep(40);

		do
		{
			clock_gettime(CLOCK_REALTIME, &tende_l );
			expiredTime = (tende_l .tv_nsec - tstart_l .tv_nsec);
		}while(expiredTime < C_SWC_SCH_TIMER_RESOLUTION);

		//Update internal clock for other software components
		swc_Scheduler_clock = swc_Scheduler_clock + (C_SWC_SCH_TIMER_RESOLUTION/1000);

		//Iterate through all possible periodic times
		for(int periodID = 0; periodID < C_SWC_SCH_NR_DIFFERENT_PERIODS; periodID++)
		{
			//Iterate through all task activation offsets (they depend on the particular core)
			for(int coreID = 0; coreID < (C_SWC_SCH_NUMBER_OF_THREADS - 1); coreID++)
			{

				if((swc_sch_task_used[coreID][periodID] != 0) &&(((operating_time - swc_sch_task_activ_offsets[coreID][periodID])% swc_sch_task_periods[periodID]) == 0))
				{
					if(operating_time >= swc_sch_task_activ_offsets[coreID][periodID])
					{
						if(swc_sch_task_event[coreID][periodID] == 0)
						{
							swc_sch_task_event[coreID][periodID] = 1;
						}
						else
						{

						}
					}


				}
			}
		}

	}

	printf("\nEnd of operation time reached -> terminate system now");
	swc_sch_terminate_threads = 1u;

}

/*************************************************************************************************
Function: 		swc_Scheduler_Thread0
Description:
*/
static void swc_Scheduler_Thread0(void)
{
	printf("\nStart Thread 0");
	fflush(stdout);


	while(swc_sch_terminate_threads == 0)
	{
		usleep(20);

		/* 20 ms Task */
		if(swc_sch_task_event[0][C_SWC_SCH_PERIOD_10MS])
		{
			swc_Sobel1_Task_periodic();

			//Reset Task Active Flag
			swc_sch_task_event[0][C_SWC_SCH_PERIOD_10MS] = 0;
		}


		/* 100 ms Task */
		if(swc_sch_task_event[0][C_SWC_SCH_PERIOD_100MS])
		{
			swc_Sobel2_Task_periodic();

			//Reset Task Active Flag
			swc_sch_task_event[0][C_SWC_SCH_PERIOD_100MS] = 0;
		}

	}


	printf("\nStop Thread0");
	fflush(stdout);
}

/*************************************************************************************************
Function: 		swc_Scheduler_Thread1
Description:
*/
static void swc_Scheduler_Thread1(void)
{
	printf("\nStart Thread 1");
	fflush(stdout);

	while(swc_sch_terminate_threads == 0)
	{
		usleep(30);

		/* 200 ms Task */
		if(swc_sch_task_event[1][C_SWC_SCH_PERIOD_200MS])
		{
			swc_MatrMul_Task_periodic();

			//Reset Task Active Flag
			swc_sch_task_event[1][C_SWC_SCH_PERIOD_200MS] = 0;
		}
	}

	printf("\nStop Thread1");
	fflush(stdout);
}

/*************************************************************************************************
Function: 		swc_Scheduler_Thread2
Description:
*/
static void swc_Scheduler_Thread2(void)
{
	printf("\nStart Thread 2");
	fflush(stdout);

	while(swc_sch_terminate_threads == 0)
	{
		GPUart_schedule();

		usleep(20);
	}

	printf("\nStop Thread2");
	fflush(stdout);
}


/*************************************************************************************************
Function: 		swc_Scheduler_MPMD
Description:	This function is called during the thread fork. It calls the assiociated function
				for each thread depending on its thread-ID.
*/
static void* swc_Scheduler_MPMD(void *args)
{
	int threadID = *((int *)args);

	struct sched_param schedulingParameter;
	schedulingParameter.__sched_priority = sched_get_priority_max(SCHED_FIFO);
	pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedulingParameter);


	switch (threadID)
	{
	case 0:
		swc_Scheduler_Thread0();
		break;

	case 1:
		swc_Scheduler_Thread1();
		break;

	case 2:
		swc_Scheduler_Thread2();
		break;
	case 3:
		/*This thread is used for real-time timing only.*/
		swc_Scheduler_Timer();
		break;
	}

	return NULL;
}


/*************************************************************************************************
Function: 		swc_Scheduler_run
Description:	This function creates the necessary number of threads for SW-Cs and joins them 
				after termination
*/

void swc_Scheduler_run(void)
{
	pthread_t threads[C_SWC_SCH_NUMBER_OF_THREADS];
	int thread_args[C_SWC_SCH_NUMBER_OF_THREADS];
	int retval;

	printf("\nStart of application...\n");
	fflush(stdout);


	for (int i = 0; i < C_SWC_SCH_NUMBER_OF_THREADS; i++)
	{
		thread_args[i] = i;
		retval = pthread_create(&threads[i], NULL, swc_Scheduler_MPMD, &thread_args[i]);
		printf("\nThread %d forked with code: %d", i, retval);
	}


	for (int i = 0; i < C_SWC_SCH_NUMBER_OF_THREADS; i++)
	{
		retval = pthread_join(threads[i], NULL);
		printf("\nThread %d joined with code: %d", i, retval);
	}
}


