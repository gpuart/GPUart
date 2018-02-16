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
* File:			SWC_Sobel1.cpp
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
#include "SWC_Sobel1.h"
#include "../../GPUart_Abstraction/GPUart_Service_IF.h"

/************************************************************************************************/
/* Constants																					*/
/************************************************************************************************/
#define C_SOB1_HEIGHT				(512)
#define C_SOB1_WIDTH				(256)

/************************************************************************************************/
/* Variables																					*/
/************************************************************************************************/
int sob1_matrix_in_s32_swc[C_SOB1_HEIGHT * C_SOB1_WIDTH];
int sob1_matrix_out_s32_swc[C_SOB1_HEIGHT * C_SOB1_WIDTH];

/************************************************************************************************/
/* Function Definition																			*/
/************************************************************************************************/


/*************************************************************************************************
Function: 		swc_Sobel1_init()
Description:	Initialize Sobel1 SW-C
*/
void swc_Sobel1_init(void)
{
	for(int i = 0; i < C_SOB1_HEIGHT; i++)
	{
		for(int j = 0; j < C_SOB1_WIDTH; j++)
		{
			if(((i >= 5)&&(i<= 20))&&((j >= 5)&&(j <= 20)))
			{
				sob1_matrix_in_s32_swc[i * C_SOB1_WIDTH + j] = 10;
			}
			else
			{
				sob1_matrix_in_s32_swc[i * C_SOB1_WIDTH + j] = 0;
			}

			if(((i >= 0)&&(i<= 3))&&((j >= 0)&&(j <= 3)))
			{
				sob1_matrix_in_s32_swc[i * C_SOB1_WIDTH + j] = 50;
			}

			sob1_matrix_out_s32_swc[i * C_SOB1_WIDTH + j] = 0;
		}
	}
}

/*************************************************************************************************
Function: 		swc_Sobel1_Task_periodic()
Description:	Run Sobel1 SW-C in 1ms period
*/
void swc_Sobel1_Task_periodic(void)
{




	if(gpuA_Sobel1_query(sob1_matrix_out_s32_swc) == GPUART_SUCCESS)
	{
//		printf("\n\nSobel1 Output:");
//		for(int i = 0; i < 25; i++)
//		{
//			printf("\n");
//			for(int j = 0; j < 25; j++)
//			{
//				printf("%d\t ", sob1_matrix_out_s32_swc[i * C_SOB1_WIDTH + j]);
//
//				sob1_matrix_out_s32_swc[i * C_SOB1_WIDTH + j] = 0;
//			}
//		}
	}
	else
	{

	}
	//Trigger Sobel1 kernel
	gpuA_Sobel1_call(sob1_matrix_in_s32_swc);



}
