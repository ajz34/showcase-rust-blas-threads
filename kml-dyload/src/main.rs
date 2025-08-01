#![allow(non_snake_case)]
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
    pub KmlGetMaxThreads: unsafe extern "C" fn() -> c_int,
    pub KmlSetNumThreads: unsafe extern "C" fn(n: c_int),
    pub BlasGetNumThreads: unsafe extern "C" fn() -> c_int,
    pub BlasSetNumThreads: unsafe extern "C" fn(n: c_int),
    pub BlasGetNumThreadsLocal: unsafe extern "C" fn() -> c_int,
    pub BlasSetNumThreadsLocal: unsafe extern "C" fn(n: c_int),
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
            let KmlGetMaxThreads = library.get(b"KmlGetMaxThreads\0").map(|sym| *sym)?;
            let KmlSetNumThreads = library.get(b"KmlSetNumThreads\0").map(|sym| *sym)?;
            let BlasGetNumThreads = library.get(b"BlasGetNumThreads\0").map(|sym| *sym)?;
            let BlasSetNumThreads = library.get(b"BlasSetNumThreads\0").map(|sym| *sym)?;
            let BlasGetNumThreadsLocal = library.get(b"BlasGetNumThreadsLocal\0").map(|sym| *sym)?;
            let BlasSetNumThreadsLocal = library.get(b"BlasSetNumThreadsLocal\0").map(|sym| *sym)?;
            let dsyevd = library.get(b"dsyevd_\0").map(|sym| *sym)?;
            Ok(Self {
                __library: library,
                dgemm,
                KmlGetMaxThreads,
                KmlSetNumThreads,
                BlasGetNumThreads,
                BlasSetNumThreads,
                BlasGetNumThreadsLocal,
                BlasSetNumThreadsLocal,
                dsyevd
            })
        }
    }
}

pub unsafe fn get_lib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| Lib::new("libklapack_full.so").unwrap())
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

fn test_inner_set_kml() {
    println!("=== Inner, KML set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { (get_lib().KmlSetNumThreads)(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { (get_lib().KmlGetMaxThreads)() };
        println!("[Thread] iter {i:2} KmlGetMaxThreads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { (get_lib().KmlGetMaxThreads)() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_set_kml_lapack() {
    println!("=== Inner, KML set lapack ===");

    let [vec_a, _, _] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { (get_lib().KmlSetNumThreads)(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { (get_lib().KmlGetMaxThreads)() };
        println!("[Thread] iter {i:2} KmlGetMaxThreads: {num_threads}");

        let mut a = vec_a[i].lock().unwrap();
        run_lapack(&mut a);
    });
    let num_threads = unsafe { (get_lib().KmlGetMaxThreads)() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_set_blas() {
    println!("=== Inner, BLAS set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { (get_lib().BlasSetNumThreadsLocal)(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { (get_lib().BlasGetNumThreadsLocal)() };
        println!("[Thread] iter {i:2} BlasGetNumThreadsLocal: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { (get_lib().BlasGetNumThreadsLocal)() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_set_blas_lapack() {
    println!("=== Inner, BLAS set ===");

    let [vec_a, _, _] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { (get_lib().BlasSetNumThreadsLocal)(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { (get_lib().BlasGetNumThreadsLocal)() };
        println!("[Thread] iter {i:2} BlasGetNumThreadsLocal: {num_threads}");

        let mut a = vec_a[i].lock().unwrap();
        run_lapack(&mut a);
    });
    let num_threads = unsafe { (get_lib().BlasGetNumThreadsLocal)() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_set_both_lapack() {
    println!("=== Inner, BLAS set ===");

    let [vec_a, _, _] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { (get_lib().BlasSetNumThreadsLocal)(1) };
        unsafe { (get_lib().KmlSetNumThreads)(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { (get_lib().BlasGetNumThreadsLocal)() };
        println!("[Thread] iter {i:2} BlasGetNumThreadsLocal: {num_threads}");
        let num_threads = unsafe { (get_lib().KmlGetMaxThreads)() };
        println!("[Thread] iter {i:2} KmlGetMaxThreads: {num_threads}");

        let mut a = vec_a[i].lock().unwrap();
        run_lapack(&mut a);
    });
    let num_threads = unsafe { (get_lib().BlasGetNumThreadsLocal)() };
    println!("[Process] threads after iteration (BlasGetNumThreadsLocal): {num_threads}");
    let num_threads = unsafe { (get_lib().KmlGetMaxThreads)() };
    println!("[Process] threads after iteration (KmlGetMaxThreads): {num_threads}");
}

fn main() {
    println!("[== KML ==]");
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

    let args = std::env::args().collect::<Vec<_>>();
    assert!(args.len() == 2);
    let mode = &args[1];

    match mode.as_str() {
        "inner-set-kml" => test_inner_set_kml(),
        "inner-set-kml-lapack" => test_inner_set_kml_lapack(),
        "inner-set-blas" => test_inner_set_blas(),
        "inner-set-blas-lapack" => test_inner_set_blas_lapack(),
        "inner-set-both-lapack" => test_inner_set_both_lapack(),
        _ => panic!("Unknown mode: {mode}"),
    }
}
