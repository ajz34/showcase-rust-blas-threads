# Test of BLAS threading controls in rust

## Results

- Outer: control thread outside rayon parallel region;
- Inner: control thread inside rayon parallel region;
- Effective: can we control the number of threads within the threshold by global number of rayon threads?
- Threads Changed: for inner case, do that changes the number of threads visible from outside of rayon parallel region?

| BLAS | Threading | Controller | Effective | Threads Changed |
|--|--|--|--|--|
| OpenBLAS | pthreads | outer `openblas_set_num_threads`       | -            | Changed |
|          |          | inner `openblas_set_num_threads`       | -            | Changed |
|          |          | outer `openblas_set_num_threads_local` | -            | Changed |
|          |          | inner `openblas_set_num_threads_local` | -            | Changed |
| OpenBLAS | OpenMP   | outer `omp_set_num_threads`            | Uncontrolled | Changed |
|          |          | inner `omp_set_num_threads`            | -            | -       |
|          |          | outer `openblas_set_num_threads`       | Uncontrolled | -       |
|          |          | inner `openblas_set_num_threads`       | Uncontrolled | -       |
|          |          | outer `openblas_set_num_threads_local` | Uncontrolled | -       |
|          |          | inner `openblas_set_num_threads_local` | Uncontrolled | -       |
| MKL      | TBB      | outer `MKL_Set_Num_Threads`            | -            | Changed |
|          |          | inner `MKL_Set_Num_Threads`            | -            | Changed |
|          |          | outer `MKL_Set_Num_Threads_Local`      | Uncontrolled | Changed |
|          |          | inner `MKL_Set_Num_Threads_Local`      | -            | -       |

- OpenBLAS with pthreads: use inner `openblas_set_num_threads` (all cases are actually the same);
- OpenBLAS with OpenMP: use inner `omp_set_num_threads`;
- MKL: use inner `MKL_Set_Num_Threads_Local`.

## Additional thoughts

- For OpenBLAS with dynamic-loading, do not hybrid use openmp (clang v.s. gnu, different gnu's). Since libopenblas.so should also linked with OpenMP runtime, so there is actually no need to explicitly specify the libgomp.so or libomp.so. 
- For MKL, the threading control function should be camel `MKL_Set_Num_Threads_Local`, instead of lower-case `mkl_set_num_threads`. Also see <https://stackoverflow.com/questions/28283112/using-mkl-set-num-threads-with-numpy>.
