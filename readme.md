

## Results

| BLAS | Threading | Controller | Effective | Threads Changed |
|--|--|--|--|--|
| OpenBLAS | pthreads | outer `openblas_set_num_threads`       | -            | Changed |
|          |          | inner `openblas_set_num_threads`       | -            | Changed |
|          |          | outer `openblas_set_num_threads_local` | -            | Changed |
|          |          | inner `openblas_set_num_threads_local` | -            | Changed |
| OpenBLAS | OpenMP   | outer `omp_set_num_threads`            | Uncontrolled | -       |
|          |          | inner `omp_set_num_threads`            | -            | -       |
|          |          | outer `openblas_set_num_threads`       | Uncontrolled | -       |
|          |          | inner `openblas_set_num_threads`       | Uncontrolled | -       |
|          |          | outer `openblas_set_num_threads_local` | Uncontrolled | -       |
|          |          | inner `openblas_set_num_threads_local` | Uncontrolled | -       |
