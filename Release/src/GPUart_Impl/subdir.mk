################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/GPUart_Impl/GPUart_Barrier.cu \
../src/GPUart_Impl/GPUart_Impl.cu \
../src/GPUart_Impl/GPUart_MatrMul.cu \
../src/GPUart_Impl/GPUart_Sobel.cu 

CU_DEPS += \
./src/GPUart_Impl/GPUart_Barrier.d \
./src/GPUart_Impl/GPUart_Impl.d \
./src/GPUart_Impl/GPUart_MatrMul.d \
./src/GPUart_Impl/GPUart_Sobel.d 

OBJS += \
./src/GPUart_Impl/GPUart_Barrier.o \
./src/GPUart_Impl/GPUart_Impl.o \
./src/GPUart_Impl/GPUart_MatrMul.o \
./src/GPUart_Impl/GPUart_Sobel.o 


# Each subdirectory must supply rules for building sources it contributes
src/GPUart_Impl/%.o: ../src/GPUart_Impl/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_53,code=sm_53 -m64 -odir "src/GPUart_Impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile --relocatable-device-code=true -gencode arch=compute_53,code=compute_53 -gencode arch=compute_53,code=sm_53 -m64 -ccbin aarch64-linux-gnu-g++  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


