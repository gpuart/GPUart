################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/SW-C/Sobel1/SWC_Sobel1.cpp 

OBJS += \
./src/SW-C/Sobel1/SWC_Sobel1.o 

CPP_DEPS += \
./src/SW-C/Sobel1/SWC_Sobel1.d 


# Each subdirectory must supply rules for building sources it contributes
src/SW-C/Sobel1/%.o: ../src/SW-C/Sobel1/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_53,code=sm_53 -m64 -odir "src/SW-C/Sobel1" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile -m64 -ccbin aarch64-linux-gnu-g++  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


