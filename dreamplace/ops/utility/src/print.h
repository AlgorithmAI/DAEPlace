#ifndef GPUPLACE_PRINT_H
#define GPUPLACE_PRINT_H

template <typename T>
void printArray(const T* x, const int n, const char* str)
{
    printf("%s[%d] = ", str, n);
    T* host_x = (T*)malloc(n * sizeof(T));
    if(host_x == NULL)
    {
        printf("failed to allocate memory on CP\n");
        return;
    }
    hipMemcpy(host_x, x, n * sizeof(T), hipMemcpyDeviceToHost);
    for(int i =0; i < n; ++i)
    {
        printf("%g ", double(host_x[i]));
    }
    printf("\n");

    free(host_x);
}

template <typename T>
void printScalar(const T& x, const char* str)
{
    printf("%s = ", str);
    T* host_x = (T*)malloc(sizeof(T));
    if(host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    hipMemcpy(host_x, &x, sizeof(T), hipMemcpyDeviceToHost);
    printf("%g\n", double(*host_x));

    free(host_x);
}

template <typename T>
void print2DArray(const T* x, const int m, const int n, const char * str)
{
    printf("%s[%dx%dx] = \n", str, m, n);
    T* host_x = (T*)malloc(m * n * sizeof(T));
    if(host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    hipMemcpy(host_x, x, m * n * sizeof(T), hipMemcpyDeviceToHost);
    for(int i = 0; i < m * m; ++i)
    {
        if(i && (i%n) == 0)
        {
            printf("\n");
        }
        printf("%g ", double(host_x[i]));
    }
    printf("\n");

    free(host_x);
}

#endif