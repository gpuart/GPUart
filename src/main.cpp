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
* File:			main.cpp
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			03.04.2017								*/
/********************************************************
*   ___   ___    ___                   ___  3     ___   *
*  |     |   |  |   |  |\  /|  |   |  |      |   |      *
*  |___  |   |  |___|  | \/ |  |   |  |___   |   |      *
*  |     |   |  |\     |    |  |   |      |  |   |      *
*  |     |___|  |  \   |    |  |___|   ___|  |   |___   *
*                                                       *
*********************************************************/

#include "SW-C/Scheduler/SWC_Scheduler.h"
#include "GPUart/GPUart.h"

#include <iostream>
#include <unistd.h>
#include <sys/resource.h>
#include <sched.h>



int main(void)
{

	int niceretval;
	niceretval = nice(-20);
	std::cout << "\nnice(-20) = "<< niceretval << std::endl;

	{
		/* Start GPUart module */
		std::cout << "***************************************************************************" << std::endl;
		std::cout << "Initialize GPUart"<< std::endl;
		GPUart_init();
		std::cout << ".. done!" << std::endl;

		/* Initialize the SW-Cs */
		std::cout << "***************************************************************************" << std::endl;
		std::cout << "Initialize SW-Cs"<< std::endl;
		swc_init();
		std::cout << ".. done!" << std::endl;

		/* Initialize the SW-C scheduler */
		std::cout << "***************************************************************************" << std::endl;
		std::cout << "Initialize SW-C Scheduler"<< std::endl;
		swc_Scheduler_init();
		std::cout << ".. done!" << std::endl;

		/* Start execution of persistent kernel */
		std::cout << "***************************************************************************" << std::endl;
		std::cout << "Start persistent kernel"<< std::endl;
		GPUart_start();
		std::cout << ".. done!" << std::endl;

		usleep(100000); //Wait 100ms

		/* Run the SW-C scheduler */
		std::cout << "***************************************************************************" << std::endl;
		std::cout << "Run SW-Components"<< std::endl;
		swc_Scheduler_run();
		std::cout << ".. done!" << std::endl;


		/* Stop execution of persistent Kernel*/
		std::cout << "***************************************************************************" << std::endl;
		std::cout << "Stop persistent kernel"<< std::endl;
		GPUart_stop();
		std::cout << ".. done!" << std::endl;

		/* Terminate GPUart module */
		std::cout << "***************************************************************************" << std::endl;
		std::cout << "Terminate GPUart"<< std::endl;
		GPUart_destroy();
		std::cout << ".. done!" << std::endl;

	}

	return 0;
}

