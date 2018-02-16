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

#ifndef GPUART_COMMON_H
#define GPUART_COMMON_H



/************************************************************************************************/
/* Include																					*/
/************************************************************************************************/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include "unistd.h"

/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/
#define C_TRUE				(1==1)
#define C_FALSE				(1==0)

#define C_UINT32_MAX		(0xFFFFFFFF)

#define C_SINT32_MAX		(0x7FFFFFFF)
#define C_SINT32_MIN		(0x10000000)
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
typedef signed int			GPUart_Retval;



/* GPGPU Language specific */


/************************************************************************************************/
/* Error Codes																					*/
/************************************************************************************************/
#define	GPUART_SUCCESS								(0x00)
#define	GPUART_NO_SUCCESS							(0x01)
#define	GPUART_ERROR_NOT_READY						(0x02)
#define GPUART_ERROR_NO_OPERTATION					(0x03)
#define GPUART_ERROR_INVALID_ARGUMENT				(0x04)
#define GPUART_ERROR_PESISTENT_KERNEL_IS_RUNNING	(0x08)
#define GPUART_ERROR_COMMAND_ALREADY_ISSUED			(0x10)


/************************************************************************************************/
/* Makro definition																				*/
/************************************************************************************************/
#define GPUART_CHECK_RETURN(value) gpuartCheckReturn(__FILE__,__LINE__, #value, value)


/************************************************************************************************/
/* Function definition																			*/
/************************************************************************************************/
static void gpuartCheckReturn(const char *file, unsigned line, const char *statement, sint32 err)
{
	if (err == 0)
		return;
	std::cerr << "GPUart RUNTIME ERROR! "<< statement << " returned " << "(" << err << ") at " << file << ":" << line << std::endl;
	//exit(1);
}

#endif
