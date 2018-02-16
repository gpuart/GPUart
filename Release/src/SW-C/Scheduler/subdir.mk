################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/SW-C/Scheduler/SWC_Scheduler.cpp 

OBJS += \
./src/SW-C/Scheduler/SWC_Scheduler.o 

CPP_DEPS += \
./src/SW-C/Scheduler/SWC_Scheduler.d 


# Each subdirectory must supply rules for building sources it contributes
src/SW-C/Scheduler/%.o: ../src/SW-C/Scheduler/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_53,code=sm_53 -m64 -odir "src/SW-C/Scheduler" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile -m64 -ccbin aarch64-linux-gnu-g++  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


