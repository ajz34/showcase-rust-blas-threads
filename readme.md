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

| BLAS | Threading | Controller | Effective | Threads Changed | LAPACK Same |
|--|--|--|--|--|--|
| OpenBLAS | pthreads | outer `openblas_set_num_threads`       | -            | Changed |
| v0.3.28  |          | inner `openblas_set_num_threads`       | -            | Changed |
|          |          | outer `openblas_set_num_threads_local` | -            | Changed |
|          |          | inner `openblas_set_num_threads_local` | -            | Changed |
| OpenBLAS | OpenMP   | outer `omp_set_num_threads`            | Uncontrolled | Changed |
| v0.3.28  |          | inner `omp_set_num_threads`            | -            | -       | Yes |
|          |          | outer `openblas_set_num_threads`       | Uncontrolled | -       |
|          |          | inner `openblas_set_num_threads`       | Uncontrolled | -       |
|          |          | outer `openblas_set_num_threads_local` | Uncontrolled | -       |
|          |          | inner `openblas_set_num_threads_local` | Uncontrolled | -       |
| MKL      | TBB      | outer `MKL_Set_Num_Threads`            | -            | Changed |
| 2025.1   |          | inner `MKL_Set_Num_Threads`            | -            | Changed |
|          |          | outer `MKL_Set_Num_Threads_Local`      | Uncontrolled | Changed |
|          |          | inner `MKL_Set_Num_Threads_Local`      | -            | -       | Yes |
| BLIS     | Any      | outer `omp_set_num_threads`            | Uncontrolled | Changed |
| v2.0     |          | inner `omp_set_num_threads`            | Uncontrolled | -       |
|          |          | outer `bli_thread_set_num_threads`     | Uncontrolled | Changed |
|          |          | inner `bli_thread_set_num_threads`     | -            | -       |
| AOCL     | -        | inner `omp_set_num_threads`            | -            | -       | Yes |
| KML      | OpenMP   | inner `KmlSetNumThreads`               | Uncontrolled | -       | Partially Controlled |
| 24.0.0   |          | inner `BlasSetNumThreads`              | -            | Changed |
|          |          | inner `BlasSetNumThreadsLocal`         | -            | -       | Uncontrolled |
|          |          | inner both `Blas...Local`/`Kml...`     | -            | -       | Yes |

- OpenBLAS with pthreads: use inner `openblas_set_num_threads` (all cases are actually the same), but note main thread is affected;
- OpenBLAS with OpenMP: use inner `omp_set_num_threads`, main thread unaffected;
- MKL: use inner `MKL_Set_Num_Threads_Local`, main thread unaffected;
- BLIS: use inner `bli_thread_set_num_threads`, main thread unaffected;
- AOCL: use inner `bli_thread_set_num_threads`, main thread unaffected;
- KML: use inner both `BlasSetNumThreadsLocal` and `KmlSetNumThreads`, main thread unaffected.

## Additional thoughts

- For OpenBLAS with dynamic-loading, do not hybrid use openmp (clang v.s. gnu, different gnu's). Since libopenblas.so should also linked with OpenMP runtime, so there is actually no need to explicitly specify the libgomp.so or libomp.so. 
- For MKL, the threading control function should be camel `MKL_Set_Num_Threads_Local`, instead of lower-case `mkl_set_num_threads`. Also see <https://stackoverflow.com/questions/28283112/using-mkl-set-num-threads-with-numpy>.
- If two variables (of type `libloading::Library`) points to the same library, one changes the mutable static variable therin (like `openblas_set_num_threads`), and then another gets the changed static variable.
