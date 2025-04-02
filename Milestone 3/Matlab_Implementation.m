%%---------------- MATLAB CUDA-INTEGRATED FWHT IMPLEMENTATION ---------------------
%% Value Initialization
N = 2^14;
fprintf("Number of Elements: %d\n", N);
x = zeros(N, 1);
for i = 0:N-1
    x(i+1) = i;
end
loop = 5;   
fprintf("Number of Loops: %d\n", loop);

%% Transfer to GPU
x_gpu = gpuArray(x);

%% CUDA-integrated FWHT
total_time = 0;
for i = 1:loop
    t0 = datetime("now");
    xr_gpu = fwht(x_gpu); % Fast Walsh-Hadamard Transform on GPU
    t1 = datetime("now");
    ms = milliseconds(t1 - t0);
    total_time = total_time + ms;
end
average_time = total_time / loop;
xr = gather(xr_gpu); % Transfer result back to CPU
fprintf('Average time elapsed (FWHT): %fms\n', average_time);

%% Output results for FWHT correction checking
fileID = fopen('fwht_output.txt','w');
for i = 1:1:size(x)
    fprintf(fileID,'%f\n',xr(i));
end
for i = 1:1:size(x)
    fprintf(fileID,'%f\n',bixr(i));
end
fclose(fileID);

%% CUDA-integrated IFWHT (Inverse Fast Walsh-Hadamard Transform)
total_time = 0;
for i = 1:loop
    t0 = datetime("now");
    y_gpu = ifwht(xr_gpu); % Compute IFWHT on GPU
    t1 = datetime("now");
    ms = milliseconds(t1 - t0);
    total_time = total_time + ms;
end
average_time = total_time / loop;
y = gather(y_gpu); % Transfer result back to CPU
fprintf('Average time elapsed (IFWHT): %fms\n', average_time);

%% Output results for IFWHT correction checking
fileID = fopen('fwht_output.txt','w');
for i = 1:1:size(x)
    fprintf(fileID,'%f\n',xr(i));
end
for i = 1:1:size(x)
    fprintf(fileID,'%f\n',bixr(i));
end
fclose(fileID);
 
