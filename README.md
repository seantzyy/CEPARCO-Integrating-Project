# CEPARCO-Integrating-Project
In Milestone 2, we were able to create the DFT and IDFT logic in C programming. In this
final project update, we will discuss the last part of the project: the implementation of FWHT and
IFWHT in CUDA. We also implemented FWHT & IFWHT in Matlab as a Built-in function to compare the speed between
Matlab and CUDA.

## Discussion of parallel algorithms implemented in our program. Highlighting which part of the sequential part is converted to parallel algorithm
In the project, FWHT uses iterative algorithms to compute it efficiently. Since it uses a bunch of repeating computation. The use of CUDA could show signs of improvement in calculating the transformation. The CUDA will implement a faster speed up for FWHT and mainly in its mostly heavy computation area of program. Mainly, the use for loop in the transformation. Another part of the project is the IFWHT, where we need to normalize the result to return the original values of the arrays. CUDA can be used to speed up the process.
![image](https://github.com/user-attachments/assets/95d74559-ccbb-43e1-8e0d-6f2544a44ac5)

In order to measure the FWHT function's execution duration, we first set up timing variables like StartingTime, EndingTime, and ElapsedMicroseconds in the CUDA environment.  Additionally, we computed the total and average execution times using QueryPerformanceFrequency() and obtained the frequency of the performance counter.  The array size (ARRAY_SIZE) was defined as a power of two, such as 2^10, 2^12, 2^14, etc., and we used cudaMallocManaged() to allocate memory appropriately.  To decide how many iterations would be carried out in order to average the findings, a loop count (loope) was set.

![image](https://github.com/user-attachments/assets/6049963f-402a-40fa-8e2e-073998035d0d)
![image](https://github.com/user-attachments/assets/52a5d0ff-7a37-4ce6-996d-7fa88e1df9b6)

After initialization, we used cudaGetDevice() to identify the GPU device being used. We then allocated memory for the following arrays:
  x: Holds the original input values.
  
  y: Stores the FWHT results.
  
  d_x: Device array for input processing.
  
  dy: Temporary array for intermediate transformations.
  
Memory allocation was done using cudaMallocManaged() with ARRAY_BYTES. We also applied cudaMemAdvise() to optimize memory access patterns and performed page creation for these arrays.

Before executing the FWHT function, we performed memory prefetching for x to ensure efficient memory access:
![image](https://github.com/user-attachments/assets/1844fd10-43e1-4f78-8e39-aa720be33a5a)

We then set up the CUDA kernel configuration where the number of elements, blocks, and threads were printed for verification. Finally, we started the program timer before launching the FWHT function.

After executing the FWHT function, we recorded the ending time and computed:
![image](https://github.com/user-attachments/assets/31c11b6f-6c3d-4701-8a31-072cd21e3bfa)

We then printed the average execution time to compare FWHT performance in CUDA and other implementations (e.g., MATLAB).
Next, we set up the Inverse FWHT (IFWHT) function by prefetching x, y, d_x, and dy and applying cudaMemAdvise() for better memory handling

The FWHT function processes the input array recursively. It accepts the following parameters:
  N: Size of the array.
  
  x: Input array.
  
  y: Output array.
  
We implemented the core FWHT logic using grid-stride loops in CUDA for efficient parallel execution:
![image](https://github.com/user-attachments/assets/390affb7-4d65-4b5c-b1e8-5c9ffb698564)

The inverse FWHT function is implemented similarly, normalizing the final results
![image](https://github.com/user-attachments/assets/9bde867b-abf4-4ca0-b43a-00f3864e33ba)

## Screenshots of Execution Time (in ms, average of 5 loops):
### A. C Sequential 
#### 2 ^ 10
![image](https://github.com/user-attachments/assets/26b56a97-e6ee-4147-9a7b-167e4609ad1e)
#### 2 ^ 12
![image](https://github.com/user-attachments/assets/1843d694-670f-489f-9878-dfb37237f4a4)
#### 2 ^ 14
![image](https://github.com/user-attachments/assets/a35840cf-1ecf-405f-8c52-1fa239260144)
#### 2 ^ 16
![image](https://github.com/user-attachments/assets/239caffe-62e2-47e7-a949-c5007fadd849)
#### 2 ^ 18 
![image](https://github.com/user-attachments/assets/e68445ca-ee92-4e98-9323-268ba78f81ff)
#### 2 ^ 20
![image](https://github.com/user-attachments/assets/9403c3cb-85a1-4572-bde2-67338e65ec4e)

### B. CUDA Integrated 
#### 2 ^ 10
![image](https://github.com/user-attachments/assets/289b013b-316c-4d54-a896-a0d220bf32d1)
#### 2 ^ 12
![image](https://github.com/user-attachments/assets/c58892ec-d804-41c3-ab0c-b83ad15cee5b)
#### 2 ^ 14
![image](https://github.com/user-attachments/assets/88c8dce6-0b9c-4e32-875f-5752b165e299)
#### 2 ^ 16
![image](https://github.com/user-attachments/assets/43cf0d9a-3487-4a22-9595-d8a3f94f9ba9)
#### 2 ^ 18
![image](https://github.com/user-attachments/assets/4b0636f0-84ad-4df4-ba33-73a180b25dae)
#### 2 ^ 20
![image](https://github.com/user-attachments/assets/4d288a39-abec-492c-9635-01a27d6462d1)

### C. MATLAB Built-in Function
#### 2 ^ 10
![image](https://github.com/user-attachments/assets/fb33f3a9-e3a1-4683-a643-83c1085af1fb)
![image](https://github.com/user-attachments/assets/e394ddb2-79a6-4a60-8fe6-73d1a742783a)
#### 2 ^ 12
![image](https://github.com/user-attachments/assets/e704661c-bfa4-4bab-ac0f-a7f9284ac22d)
![image](https://github.com/user-attachments/assets/ba45a3c0-a588-46f9-b03f-154149639332)
#### 2 ^ 14
![image](https://github.com/user-attachments/assets/7569b2e0-15f7-487a-8776-1443e4ffc4e1)
![image](https://github.com/user-attachments/assets/99673904-2450-4d19-8b34-63b430feadb2)
#### 2 ^ 16
![image](https://github.com/user-attachments/assets/153a2f2c-6de9-416f-a51b-923aa58116aa)
![image](https://github.com/user-attachments/assets/31e1f168-027e-4076-b4ba-66e744e5686a)

## Comparison Table for FWHT
|  | **Sequential C** | **CUDA** | **MATLAB** |
| ------------- | ------------- | ------------- | ------------- |
| 2^10 | 0.0288 mS | 1.104  | 782.7536 | 
| 2^12 | 0.1256 mS | 0.6858  | 3,344.8146 | 
| 2^14 | 0.549 mS | 0.9244  | 15551.295 | 
| 2^16 | 2.451 mS | 1.0738  | 74488.5234 | 
| 2^18 | 10.3672 mS | 2.2194| N/A | 
| 2^20 | 44.0258 mS | 8.5298  | N/A | 

## Comparison Table for IFWHT
|  | **Sequential C** | **CUDA** | **MATLAB** |
| ------------- | ------------- | ------------- | ------------- |
| 2^10 | 0.4 mS | 0.3172  | 656.433 | 
| 2^12 | 0.1692 mS | 0.5378  | 3,345.1854 | 
| 2^14 | 0.767 mS | 0.6316  | 15,610.7164 | 
| 2^16 | 3.161 mS | 1.095  | 77,413.7126 | 
| 2^18 | 14.4688 mS | 2.3312  | N/A | 
| 2^20 | 56.3722 mS | 8.771  | N/A | 

## Comparing Performance through Execution Time
### Using total execution time and Arithmetic Mean
|  | **Sequential C** | **CUDA** | **MATLAB** |
| ------------- | ------------- | ------------- | ------------- |
| 2^10 | 0.0288 mS | 1.104  | 782.7536 | 
| 2^12 | 0.1256 mS | 0.6858  | 3,344.8146 | 
| 2^14 | 0.549 mS | 0.9244  | 15,551.295 | 
| 2^16 | 2.451 mS | 1.0738  | 74,488.5234 | 
| 2^18 | 10.3672 mS | 2.2194| N/A | 
| 2^20 | 44.0258 mS | 8.5298  | N/A | 
| Total Time | 55.0964 mS | 14.5372 mS | 94,167.3866 mS | 
| Arithmetic Mean | 9.1827 mS | 2.4229 mS | 15,694 mS | 

|  | **Sequential C** | **CUDA** | **MATLAB** |
| ------------- | ------------- | ------------- | ------------- |
| 2^10 | 0.4 mS | 0.3172  | 656.433 | 
| 2^12 | 0.1692 mS | 0.5378  | 3,345.1854 | 
| 2^14 | 0.767 mS | 0.6316  | 15,610.7164 | 
| 2^16 | 3.161 mS | 1.095  | 77,413.7126 | 
| 2^18 | 14.4688 mS | 2.3312  | N/A | 
| 2^20 | 56.3722 mS | 8.771  | N/A | 
| Total Time | 75.3382 mS | 13.0522 mS | 97,026.0474 mS | 
| Arithmetic Mean | 12.5564 mS | 2.1754 mS | 16,171.0079 mS | 

## Error Checking 
For correctness checking, we output the results from our Matlab implementation and
Matlab built-in functions. The results are stored in text files, which is then compared to the
results in CUDA. We error-checked Matlab vs our C++ FWHT implementation. We error-checked for 
2^10 until 2^20. We compared both of the output text files and concluded that our implementation 
in C++ along with CUDA is correct. 
![image](https://github.com/user-attachments/assets/c5361795-32c2-448f-b6c5-ad0694cf6d3e)

## Analysis
