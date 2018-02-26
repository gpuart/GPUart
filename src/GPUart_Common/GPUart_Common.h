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

/*!	@file 	GPUart_Common.h
 *
 * 	@brief 	A library which defines some common macros, typedefs.
 *
 *			Implements the service-orientated init-, call-, and query-interfaces, respectively for each kernel.
 * 			Call a kernel's init-interface to initialize kernel related GPU data.
 * 			Call a kernel's call-interface to enqueue a new kernel instance in the scheduler and update GPU data.
 * 			Call a kernel's query-interface to query kernel completion and to get the output of the kernel.
 * 			This layer is used to achieve higher portability by abstracting the systems's heterogeneity.
 *
 * 	@author	Christoph Hartmann
 *  @date	Created on: 3 Apr 2017
 */

#ifndef GPUART_COMMON_H
#define GPUART_COMMON_H



/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/
#include <stdio.h>	//Standard I/O
#include <iostream>	//Standard I/O


/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/

/*!	@def C_TRUE
 *  @brief TRUE condition definition
 */
#define C_TRUE				(1==1)

/*!	@def C_FALSE
 *	@brief FALSE condition definition
 */
#define C_FALSE				(1==0)

/************************************************************************************************/
/* Typedefs																						*/
/************************************************************************************************/

/* Native datatypes*/
typedef	signed int			sint32;
typedef unsigned int		uint32;

typedef signed char			sint8;
typedef unsigned char		uint8;

typedef signed short		sint16;
typedef unsigned short		uint16;

typedef signed long long	sint64;
typedef unsigned long long	uint64;

typedef float				float32;
typedef double				float64;

/* Specific datatypes */

/*!	@typedef GPUart_Retval
 *	@brief The standard return type of functions in GPUart, representing an error code.
 */
typedef signed int			GPUart_Retval;



/* GPGPU Language specific */


/************************************************************************************************/
/* Error Codes																					*/
/************************************************************************************************/

/*!	@def GPUART_SUCCESS
 *	@brief Error code 0x00 -> Function returned successfully
 */
/*!	@def GPUART_NO_SUCCESS
 *	@brief Error code 0x01 -> Function returned unsuccessfully
 */
/*!	@def GPUART_ERROR_NOT_READY
 *	@brief Error code 0x02 -> Error: operation not ready
 */
/*!	@def GPUART_ERROR_NO_OPERTATION
 *	@brief Error code 0x03 -> Error: function return without doing anything
 */
/*!	@def GPUART_ERROR_INVALID_ARGUMENT
 *	@brief Error code 0x04 -> Error: an invalid function parameter has been used
 */
/*!	@def GPUART_ERROR_PESISTENT_KERNEL_IS_RUNNING
 *	@brief Error code 0x03 -> Error: operation could not be executed, since the
 *									 persistent GPU thread is already running.
 */

#define	GPUART_SUCCESS								(0x00)
#define	GPUART_NO_SUCCESS							(0x01)
#define	GPUART_ERROR_NOT_READY						(0x02)
#define GPUART_ERROR_NO_OPERTATION					(0x03)
#define GPUART_ERROR_INVALID_ARGUMENT				(0x04)
#define GPUART_ERROR_PESISTENT_KERNEL_IS_RUNNING	(0x08)



/************************************************************************************************/
/* Macro definition																				*/
/************************************************************************************************/

/*!	@def GPUART_CHECK_RETURN
 *	@brief Standard macro for error checking within GPUart.
 */
#define GPUART_CHECK_RETURN(value) gpuartCheckReturn(__FILE__,__LINE__, #value, value)


/************************************************************************************************/
/* Function definition																			*/
/************************************************************************************************/


/*!	@brief Standard function for error checking within GPUart.
 *
 */
static void gpuartCheckReturn(const char *file, unsigned line, const char *statement, sint32 err)
{
	if (err == 0)
		return;
	std::cerr << "GPUart RUNTIME ERROR! "<< statement << " returned " << "(" << err << ") at " << file << ":" << line << std::endl;
	//exit(1);
}

#endif
