#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use libloading::Library;
use rayon::prelude::*;
use std::ffi::{c_char, c_int};
use std::sync::Mutex;

pub struct Lib {
    __library: Library,
    pub dgemm: unsafe extern "C" fn(
        transa: *mut c_char,
        transb: *mut c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f64,
        a: *const f64,
        lda: *const c_int,
        b: *const f64,
        ldb: *const c_int,
        beta: *const f64,
        c: *mut f64,
        ldc: *const c_int,
    ),
    pub mkl_get_max_threads: unsafe extern "C" fn() -> c_int,
    pub mkl_set_num_threads: unsafe extern "C" fn(n: c_int),
    pub mkl_set_num_threads_local: unsafe extern "C" fn(n: c_int),
    pub dsyevd: unsafe extern "C" fn(
        jobz: *const c_char,
        uplo: *const c_char,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        w: *mut f64,
        work: *mut f64,
        lwork: *mut c_int,
        iwork: *mut c_int,
        liwork: *mut c_int,
        info: *mut c_int,
    ),
}

impl Lib {
    pub unsafe fn new<P>(path: P) -> Result<Self, ::libloading::Error>
    where
        P: AsRef<::std::ffi::OsStr>,
    {
        let library = ::libloading::Library::new(path)?;
        Self::from_library(library)
    }

    pub fn from_library(library: Library) -> Result<Self, ::libloading::Error> {
        unsafe {
            let dgemm = library.get(b"dgemm_\0").map(|sym| *sym)?;
            let mkl_get_max_threads = library.get(b"MKL_Get_Max_Threads\0").map(|sym| *sym)?;
            let mkl_set_num_threads = library.get(b"MKL_Set_Num_Threads\0").map(|sym| *sym)?;
            let mkl_set_num_threads_local = library.get(b"MKL_Set_Num_Threads_Local\0").map(|sym| *sym)?;
            let dsyevd = library.get(b"dsyevd_\0").map(|sym| *sym)?;
            Ok(Self {
                __library: library,
                dgemm,
                mkl_get_max_threads,
                mkl_set_num_threads,
                mkl_set_num_threads_local,
                dsyevd,
            })
        }
    }
}

pub unsafe fn get_lib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| Lib::new("libmkl_rt.so").unwrap())
}

pub unsafe fn dgemm(
    transa: *mut c_char,
    transb: *mut c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const f64,
    a: *const f64,
    lda: *const c_int,
    b: *const f64,
    ldb: *const c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: *const c_int,
) {
    (get_lib().dgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

pub unsafe fn dsyevd(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const c_int,
    a: *mut f64,
    lda: *const c_int,
    w: *mut f64,
    work: *mut f64,
    lwork: *mut c_int,
    iwork: *mut c_int,
    liwork: *mut c_int,
    info: *mut c_int,
) {
    (get_lib().dsyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
}

pub unsafe fn mkl_get_max_threads() -> c_int {
    (get_lib().mkl_get_max_threads)()
}

pub unsafe fn mkl_set_num_threads(n: c_int) {
    (get_lib().mkl_set_num_threads)(n);
}

pub unsafe fn mkl_set_num_threads_local(n: c_int) {
    (get_lib().mkl_set_num_threads_local)(n);
}

fn gen_vecs() -> [Vec<Mutex<Vec<f64>>>; 3] {
    let vec_a: Vec<Mutex<Vec<f64>>> =
        (0..16).map(|_| Mutex::new((0..2048 * 2048).map(|x| x as f64 / 2048.0).collect())).collect();
    let vec_b: Vec<Mutex<Vec<f64>>> =
        (0..16).map(|_| Mutex::new((0..2048 * 2048).map(|x| x as f64 / 2048.0).collect())).collect();
    let vec_c: Vec<Mutex<Vec<f64>>> =
        (0..16).map(|_| Mutex::new((0..2048 * 2048).map(|x| x as f64 / 2048.0).collect())).collect();
    [vec_a, vec_b, vec_c]
}

fn run_blas(a: &[f64], b: &[f64], c: &mut [f64]) {
    let n = 2048;
    let mut stat_p = perf_monitor::cpu::ProcessStat::cur().unwrap();
    let time = std::time::Instant::now();
    unsafe {
        let t_char = b"T";
        let n_char = b"N";
        dgemm(
            t_char.as_ptr() as *mut c_char,
            n_char.as_ptr() as *mut c_char,
            &n,
            &n,
            &n,
            &3.0,
            a.as_ptr(),
            &n,
            b.as_ptr(),
            &n,
            &0.0,
            c.as_mut_ptr(),
            &n,
        );
    }
    let elapsed = time.elapsed();
    let usage_p = stat_p.cpu().unwrap() * 100.0;
    let thread_id = rayon::current_thread_index().unwrap_or(0);
    println!("[CPU] rayon thread id {thread_id:2}, wall time: {elapsed:8.2?}, process usage: {usage_p:.2}%");
}

fn run_lapack(a: &mut [f64]) {
    let n = 2048;
    let mut stat_p = perf_monitor::cpu::ProcessStat::cur().unwrap();
    let time = std::time::Instant::now();
    unsafe {
        let jobz = b"V";
        let uplo = b"U";
        let mut w: Vec<f64> = vec![0.0; n];
        let lwork = 2 * (2 * n * n + 6 * n + 1);
        let liwork = 2 * (5 * n + 3);
        let mut work: Vec<f64> = vec![0.0; lwork];
        let mut iwork: Vec<i32> = vec![0; liwork];
        let mut info: i32 = 0;

        dsyevd(
            jobz.as_ptr() as *const c_char,
            uplo.as_ptr() as *const c_char,
            &(n as _),
            a.as_mut_ptr(),
            &(n as _),
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &mut (lwork as _),
            iwork.as_mut_ptr(),
            &mut (liwork as _),
            &mut info,
        );
    }
    let elapsed = time.elapsed();
    let usage_p = stat_p.cpu().unwrap() * 100.0;
    let thread_id = rayon::current_thread_index().unwrap_or(0);
    println!("[LAPACK] rayon thread id {thread_id:2}, wall time: {elapsed:8.2?}, process usage: {usage_p:.2}%");
}

fn test_outer_set() {
    println!("=== Outer, set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    unsafe { mkl_set_num_threads(1) };
    (0..16).into_par_iter().for_each(|i| {
        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { mkl_get_max_threads() };
        println!("[Thread] iter {i:2} mkl_get_max_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { mkl_get_max_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_set() {
    println!("=== Inner, set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { mkl_set_num_threads(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { mkl_get_max_threads() };
        println!("[Thread] iter {i:2} mkl_get_max_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { mkl_get_max_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_outer_set_local() {
    println!("=== Outer, set_local ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    unsafe { mkl_set_num_threads_local(1) };
    (0..16).into_par_iter().for_each(|i| {
        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { mkl_get_max_threads() };
        println!("[Thread] iter {i:2} mkl_get_max_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { mkl_get_max_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_set_local() {
    println!("=== Inner, set_local ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { mkl_set_num_threads_local(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { mkl_get_max_threads() };
        println!("[Thread] iter {i:2} mkl_get_max_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { mkl_get_max_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_set_local_lapack() {
    println!("=== Inner, set_local ===");

    let [vec_a, _, _] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { mkl_set_num_threads_local(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { mkl_get_max_threads() };
        println!("[Thread] iter {i:2} mkl_get_max_threads: {num_threads}");

        let mut a = vec_a[i].lock().unwrap();
        run_lapack(&mut a);
    });
    let num_threads = unsafe { mkl_get_max_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn main() {
    println!("[== MKL ==]");
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

    let args = std::env::args().collect::<Vec<_>>();
    assert!(args.len() == 2);
    let mode = &args[1];

    match mode.as_str() {
        "outer-set" => test_outer_set(),
        "inner-set" => test_inner_set(),
        "outer-set-local" => test_outer_set_local(),
        "inner-set-local" => test_inner_set_local(),
        "inner-set-local-lapack" => test_inner_set_local_lapack(),
        _ => panic!("Unknown mode: {mode}"),
    }
}
