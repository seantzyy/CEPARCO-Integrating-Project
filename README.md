# CEPARCO-Integrating-Project
In Milestone 2, we were able to create the DFT and IDFT logic in C programming. In this
final project update, we will discuss the last part of the project: the implementation of FWHT and
IFWHT in CUDA. We also implemented FWHT & IFWHT in Matlab to compare the speed between
Matlab and CUDA. Lastly, we tried using the DFT & IDFT built-in functions in Matlab, and see
if it is faster than CUDA. 

## Comparison of Execution Time (in ms, average of 5 loops):
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

| **Sequential C** | **CUDA** | **MATLAB** |
| ------------- | ------------- | ------------- |

## Error Checking 
For correctness checking, we output the results from our Matlab implementation and
Matlab built-in functions. The results are stored in text files, which is then compared to the
results in CUDA. We error-checked Matlab vs our C++ FWHT implementation. We error-checked for 
2^10 until 2^20. We compared both of the output text files and concluded that our implementation 
in C++ along with CUDA is correct. 
![image](https://github.com/user-attachments/assets/c5361795-32c2-448f-b6c5-ad0694cf6d3e)

