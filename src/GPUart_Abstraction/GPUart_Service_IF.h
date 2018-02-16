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
* File:			GPUart_Service_IF.h
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

#ifndef GPUART_SERVICE_IF_H
#define GPUART_SERVICE_IF_H


/************************************************************************************************/
/* Include																						*/
/************************************************************************************************/
#include "../GPUart_Common/GPUart_Common.h"



/************************************************************************************************/
/* Function declaration																			*/
/************************************************************************************************/




GPUart_Retval gpuA_Sobel1_init
(
		void
);

GPUart_Retval gpuA_Sobel1_call
(
		sint32* sob1_matrix_in_s32_swc
);

GPUart_Retval gpuA_Sobel1_query
(
		sint32* sob1_matrix_out_s32_swc
);


GPUart_Retval gpuA_Sobel2_init
(
		void
);

GPUart_Retval gpuA_Sobel2_call
(
		sint32* sob2_matrix_in_s32_swc
);

GPUart_Retval gpuA_Sobel2_query
(
		sint32* sob2_matrix_out_s32_swc
);




GPUart_Retval gpuA_MM_init
(
		void
);

GPUart_Retval gpuA_MM_call
(
		float32* mm_MatrixA_f32_swc,
		float32* mm_MatrixB_f32_swc
);

GPUart_Retval gpuA_MM_query
(
		float32* mm_MatrixC_f32_swc
);



#endif
