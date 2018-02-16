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
* File:			SWC_MM.cpp
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			13.10.2017								*/
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
#include "SWC_MM.h"
#include "../../GPUart_Abstraction/GPUart_Service_IF.h"

/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/
#define C_MM_MATRIX_N				(768)

/************************************************************************************************/
/* Variables																					*/
/************************************************************************************************/
float mm_MatrixA_f32_swc[C_MM_MATRIX_N * C_MM_MATRIX_N];
float mm_MatrixB_f32_swc[C_MM_MATRIX_N * C_MM_MATRIX_N];
float mm_MatrixC_f32_swc[C_MM_MATRIX_N * C_MM_MATRIX_N];

/************************************************************************************************/
/* Function Definition																			*/
/************************************************************************************************/


/*************************************************************************************************
Function: 		swc_MatrMul_init()
Description:	Initialize MatrMul SW-C
*/
void swc_MatrMul_init(void)
{
	for(int i = 0; i < C_MM_MATRIX_N; i++)
	{
		for(int j = 0; j < C_MM_MATRIX_N; j++)
		{
			if(j == 0)
			{
				mm_MatrixA_f32_swc[i * C_MM_MATRIX_N + j] = 3.0f;
				mm_MatrixB_f32_swc[i * C_MM_MATRIX_N + j] = 3.0f;
			}
			else if(i == 0)
			{
				mm_MatrixA_f32_swc[i * C_MM_MATRIX_N + j] = 2.0f;
				mm_MatrixB_f32_swc[i * C_MM_MATRIX_N + j] = 2.0f;
			}
			else
			{
				mm_MatrixA_f32_swc[i * C_MM_MATRIX_N + j] = 1.0f;
				mm_MatrixB_f32_swc[i * C_MM_MATRIX_N + j] = 1.0f;
			}

			mm_MatrixC_f32_swc[i * C_MM_MATRIX_N + j] = 0;
		}
	}
}

/*************************************************************************************************
Function: 		swc_MM_Task_periodic()
Description:	Run MatrMul SW-C in 1000ms period
*/
void swc_MatrMul_Task_periodic(void)
{




	if(gpuA_MM_query(mm_MatrixC_f32_swc) == GPUART_SUCCESS)
	{
//		printf("Print Matrix\n");
//		for(int i = 0; i < 20; i++)
//		{
//			for(int j = 0; j < 20; j++)
//			{
//				printf("%.3f \t ", mm_MatrixC_f32_swc[i * C_MM_MATRIX_N + j]);
//			}
//			printf("\n");
//		}
	}
	else
	{

	}
	//Trigger MM kernel
	gpuA_MM_call(mm_MatrixA_f32_swc, mm_MatrixB_f32_swc);



}
