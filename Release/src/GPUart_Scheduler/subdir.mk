################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/GPUart_Scheduler/GPUart_Scheduler.cpp 

OBJS += \
./src/GPUart_Scheduler/GPUart_Scheduler.o 

CPP_DEPS += \
./src/GPUart_Scheduler/GPUart_Scheduler.d 


# Each subdirectory must supply rules for building sources it contributes
src/GPUart_Scheduler/%.o: ../src/GPUart_Scheduler/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_53,code=sm_53 -m64 -odir "src/GPUart_Scheduler" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile -m64 -ccbin aarch64-linux-gnu-g++  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


