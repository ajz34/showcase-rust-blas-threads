# Test of BLAS threading controls in rust

This showcase repo tests BLAS threading control in rust:
- my PC has 16 cores in total;
- confine 4 available threads for rust's rayon;
- we should expect that only 400% CPU usage when using multi-threaded BLAS;
- how to confine: control number of threads to exactly 1 in rayon parallel region.

## Results

- Outer: control thread outside rayon parallel region;
- Inner: control thread inside rayon parallel region;
- Effective: can we control the number of threads within the threshold by global number of rayon threads?
- Threads Changed: for inner case, do that changes the number of threads visible from outside of rayon parallel region?

| BLAS | Threading | Controller | Effective | Threads Changed |
|--|--|--|--|--|
| OpenBLAS | pthreads | outer `openblas_set_num_threads`       | -            | Changed |
| v0.3.28  |          | inner `openblas_set_num_threads`       | -            | Changed |
|          |          | outer `openblas_set_num_threads_local` | -            | Changed |
|          |          | inner `openblas_set_num_threads_local` | -            | Changed |
| OpenBLAS | OpenMP   | outer `omp_set_num_threads`            | Uncontrolled | Changed |
| v0.3.28  |          | inner `omp_set_num_threads`            | -            | -       |
|          |          | outer `openblas_set_num_threads`       | Uncontrolled | -       |
|          |          | inner `openblas_set_num_threads`       | Uncontrolled | -       |
|          |          | outer `openblas_set_num_threads_local` | Uncontrolled | -       |
|          |          | inner `openblas_set_num_threads_local` | Uncontrolled | -       |
| MKL      | TBB      | outer `MKL_Set_Num_Threads`            | -            | Changed |
| 2025.1   |          | inner `MKL_Set_Num_Threads`            | -            | Changed |
|          |          | outer `MKL_Set_Num_Threads_Local`      | Uncontrolled | Changed |
|          |          | inner `MKL_Set_Num_Threads_Local`      | -            | -       |
| BLIS     | Any      | outer `omp_set_num_threads`            | Uncontrolled | -       |
| v2.0     |          | inner `omp_set_num_threads`            | Uncontrolled | -       |
|          |          | outer `bli_thread_set_num_threads`     | Uncontrolled | Changed |
|          |          | inner `bli_thread_set_num_threads`     | -            | -       |

- OpenBLAS with pthreads: use inner `openblas_set_num_threads` (all cases are actually the same);
- OpenBLAS with OpenMP: use inner `omp_set_num_threads`;
- MKL: use inner `MKL_Set_Num_Threads_Local`;
- BLIS: use inner `bli_thread_set_num_threads`.

## Additional thoughts

- For OpenBLAS with dynamic-loading, do not hybrid use openmp (clang v.s. gnu, different gnu's). Since libopenblas.so should also linked with OpenMP runtime, so there is actually no need to explicitly specify the libgomp.so or libomp.so. 
- For MKL, the threading control function should be camel `MKL_Set_Num_Threads_Local`, instead of lower-case `mkl_set_num_threads`. Also see <https://stackoverflow.com/questions/28283112/using-mkl-set-num-threads-with-numpy>.
