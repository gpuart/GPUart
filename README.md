Welcome to the prototype of GPUart, an application-based limited preemtive GPU real-time scheduler, focusing embedded systems and integrated GPUs.
In order to schedule kernel real-time capable, GPUart includes a light-weight high-level GPU resource managment approach, 
enabling Gang scheduling on GPUs. GPUart can preempt kernel which implement our approach for limited preemption on instruction level,
according to the fixed preemption point strategy.
GPUart is placed on top of the GPGPU driver stack and requires neither hardware- nor driver stack extensions to schedule
GPGPU kernel in real-time fashion, instead, it uses only state-of-the-art GPU hardware- and software features. 

Our journal paper about GPUart has recently been submitted for publication. The link to the paper will be provided as soon as the 
paper has been released. 

Please see the copyright notice in the file COPYRIGHT.txt, which is distributed in the same directory as this file.

This version of GPUart has been tested with CUDA version 8.0.

Contributions:
==============

GPUart was created by Christoph A. Hartmann, a research scientist in the research group of Ulrich Margull at the Technische Hochschule Ingolstadt
(THI, engl. Technical University of Applied Science Ingolstadt). GPUart was developed within the scope of the research project FORMUS³IC (www.formus3ic.de),
founded by the Bayerische Forschungsstiftung (BFS, engl. Bavarian Research Foundation). 


General Hints:
==============

	* The source code of GPUart is implemented in the folders GPUart, GPUart_Abstraction, GPUart_Common, GPUart_Config, GPUart_Impl, and GPUart_Scheduler. 
	  This version of GPUart implements two examplary kernels, namely a Matrix Multiplication and a Sobel Filter. 
	  The folder SW-C contains the "Host"-code. This code implements no features of GPUart, instead, it uses GPUart to schedule GPGPU kernels real-time capable. 

	* GPUart focuses on integrated GPUs. This version of GPUart does not implement memory transfers between descrete GPUs and CPUs.

	* This version of GPUart uses CUDA v8.0. An OpenCL version is currently not implemented.

Configuration Hints:
====================

	* For preemptive scheduling, comment line 41 in GPUart_Scheduler/GPUart_Scheduler.cpp (//#define S_NON_PREEMPTIVE) 
	  and uncomment line 28 in GPUart_Impl/GPUart_Impl.cuh (#define S_MEAS_PREEMPTIV)

	* For non-preemptive scheduling, uncomment line 41 in GPUart_Scheduler/GPUart_Scheduler.cpp (#define S_NON_PREEMPTIVE) 
	  and comment line 28 in GPUart_Impl/GPUart_Impl.cuh (//#define S_MEAS_PREEMPTIV)

	* The scheduling policy of GPUart (Gang-EDF or Gang-FTP/DM) can be configured in lines 33-37 in 
	  GPUart_Scheduler/GPUart_Scheduler.cpp.

	* The activation of kernel tasks is implemented in SW-C/Scheduler/SWC_Scheduler.cpp. Tasks' periods and offsets are 
	  defined in this file, too.

	* The deadline of a kernel is specified in the array gpuS_relDeadlines_u32 (line 100-104 in 
	  GPUart_Scheduler/GPUart_Scheduler.cpp). 

