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
* File:			SWC_Sobel2.cpp
* Created by: 	Christoph Hartmann
* Institute:	Technische Hochschule Ingolstadt
* Date:			18.05.2017								*/
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
#include "SWC_Sobel2.h"
#include "../../GPUart_Abstraction/GPUart_Service_IF.h"

/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/
#define C_SOB2_HEIGHT				(256)
#define C_SOB2_WIDTH				(1024)

/************************************************************************************************/
/* Variables																					*/
/************************************************************************************************/
int sob2_matrix_in_s32_swc[C_SOB2_HEIGHT * C_SOB2_WIDTH];
int sob2_matrix_out_s32_swc[C_SOB2_HEIGHT * C_SOB2_WIDTH];

/************************************************************************************************/
/* Function Definition																			*/
/************************************************************************************************/


/*************************************************************************************************
Function: 		swc_Sobel2_init()
Description:	Initialize Sobel2 SW-C
*/
void swc_Sobel2_init(void)
{
	for(int i = 0; i < C_SOB2_HEIGHT; i++)
	{
		for(int j = 0; j < C_SOB2_WIDTH; j++)
		{
			if(((i >= 0)&&(i<= 10))&&((j >= 10)&&(j <= 15)))
			{
				sob2_matrix_in_s32_swc[i * C_SOB2_WIDTH + j] = 10;
			}
			else
			{
				sob2_matrix_in_s32_swc[i * C_SOB2_WIDTH + j] = 0;
			}


			sob2_matrix_out_s32_swc[i * C_SOB2_WIDTH + j] = 0;
		}
	}
}

/*************************************************************************************************
Function: 		swc_Sobel2_Task_periodic()
Description:	Run Sobel2 SW-C in period
*/
void swc_Sobel2_Task_periodic(void)
{


	if(gpuA_Sobel2_query(sob2_matrix_out_s32_swc) == GPUART_SUCCESS)
	{
//		printf("\n\nSobel2 Output:");
//		for(int i = 0; i < 25; i++)
//		{
//			printf("\n");
//			for(int j = 0; j < 25; j++)
//			{
//				printf("%d\t ", sob2_matrix_out_s32_swc[i * C_SOB2_WIDTH + j]);
//
//				sob2_matrix_out_s32_swc[i * C_SOB2_WIDTH + j] = 0;
//			}
//		}
	}
	else
	{

	}

	//Trigger Sobel2 kernel
	gpuA_Sobel2_call(sob2_matrix_in_s32_swc);



}
