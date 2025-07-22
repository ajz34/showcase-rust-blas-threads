use blas::dgemm;
use rayon::prelude::*;
use std::sync::Mutex;

unsafe extern "C" {
    pub fn omp_get_num_threads() -> ::std::os::raw::c_int;
    pub fn omp_get_max_threads() -> ::std::os::raw::c_int;
    pub fn omp_set_num_threads(n: ::std::os::raw::c_int);
    pub fn openblas_set_num_threads(n: ::std::os::raw::c_int);
    pub fn openblas_get_num_threads() -> ::std::os::raw::c_int;
    pub fn openblas_set_num_threads_local(n: ::std::os::raw::c_int);
    pub fn openblas_get_config() -> *mut ::std::os::raw::c_char;
}

fn gen_vecs() -> [Vec<Mutex<Vec<f64>>>; 3] {
    let vec_a: Vec<Mutex<Vec<f64>>> =
        (0..16).map(|_| Mutex::new((0..1024 * 1024).map(|x| x as f64 / 1024.0).collect())).collect();
    let vec_b: Vec<Mutex<Vec<f64>>> =
        (0..16).map(|_| Mutex::new((0..1024 * 1024).map(|x| x as f64 / 1024.0).collect())).collect();
    let vec_c: Vec<Mutex<Vec<f64>>> =
        (0..16).map(|_| Mutex::new((0..1024 * 1024).map(|x| x as f64 / 1024.0).collect())).collect();
    [vec_a, vec_b, vec_c]
}

fn run_blas(a: &[f64], b: &[f64], c: &mut [f64]) {
    let n = 1024;
    let mut stat_p = perf_monitor::cpu::ProcessStat::cur().unwrap();
    let time = std::time::Instant::now();
    unsafe {
        dgemm(b'T', b'N', n, n, n, 3.0, a, n, b, n, 0.0, c, n);
    }
    let elapsed = time.elapsed();
    let usage_p = stat_p.cpu().unwrap() * 100.0;
    let thread_id = rayon::current_thread_index().unwrap_or(0);
    println!("[CPU] rayon thread id {thread_id:2}, wall time: {elapsed:8.2?}, process usage: {usage_p:.2}%");
}

fn test_outer_openblas_set() {
    println!("=== Outer, OpenBLAS set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    unsafe { openblas_set_num_threads(1) };
    (0..16).into_par_iter().for_each(|i| {
        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { openblas_get_num_threads() };
        println!("[Thread] iter {i:2} openblas_get_num_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { openblas_get_num_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_openblas_set() {
    println!("=== Inner, OpenBLAS set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { openblas_set_num_threads(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { openblas_get_num_threads() };
        println!("[Thread] iter {i:2} openblas_get_num_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { openblas_get_num_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_outer_openblas_set_local() {
    println!("=== Outer, OpenBLAS set local ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    unsafe { openblas_set_num_threads_local(1) };
    (0..16).into_par_iter().for_each(|i| {
        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { openblas_get_num_threads() };
        println!("[Thread] iter {i:2} openblas_get_num_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { openblas_get_num_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_openblas_set_local() {
    println!("=== Inner, OpenBLAS set local ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { openblas_set_num_threads_local(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { openblas_get_num_threads() };
        println!("[Thread] iter {i:2} openblas_get_num_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { openblas_get_num_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn main() {
    println!("[== OpenBLAS pthreads ==]");
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

    // print OpenBLAS configuration
    unsafe {
        let config = openblas_get_config();
        let config_str = std::ffi::CStr::from_ptr(config).to_string_lossy();
        println!("OpenBLAS configuration: {config_str}");
    }

    test_outer_openblas_set();
    test_inner_openblas_set();
    test_outer_openblas_set_local();
    test_inner_openblas_set_local();
}
