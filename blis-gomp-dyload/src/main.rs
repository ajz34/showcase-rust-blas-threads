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
    pub bli_thread_set_num_threads: unsafe extern "C" fn(num: i32),
    pub bli_thread_get_num_threads: unsafe extern "C" fn() -> i32,
    pub omp_get_max_threads: unsafe extern "C" fn() -> c_int,
    pub omp_set_num_threads: unsafe extern "C" fn(n: c_int),
}

impl Lib {
    pub unsafe fn new<P>(path: P, gomp_path: P) -> Result<Self, ::libloading::Error>
    where
        P: AsRef<::std::ffi::OsStr>,
    {
        let library = ::libloading::Library::new(path)?;
        let gomp_library = ::libloading::Library::new(gomp_path)?;
        Self::from_library(library, gomp_library)
    }

    pub fn from_library(library: Library, gomp_library: Library) -> Result<Self, ::libloading::Error> {
        unsafe {
            let dgemm = library.get(b"dgemm_\0").map(|sym| *sym)?;
            let bli_thread_set_num_threads = library.get(b"bli_thread_set_num_threads\0").map(|sym| *sym)?;
            let bli_thread_get_num_threads = library.get(b"bli_thread_get_num_threads\0").map(|sym| *sym)?;
            let omp_get_max_threads = gomp_library.get(b"omp_get_max_threads\0").map(|sym| *sym)?;
            let omp_set_num_threads = gomp_library.get(b"omp_set_num_threads\0").map(|sym| *sym)?;
            Ok(Self {
                __library: library,
                dgemm,
                bli_thread_set_num_threads,
                bli_thread_get_num_threads,
                omp_get_max_threads,
                omp_set_num_threads,
            })
        }
    }
}

pub unsafe fn get_lib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| Lib::new("libblis.so", "libgomp.so").unwrap())
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

pub unsafe fn bli_thread_set_num_threads(num: i32) {
    (get_lib().bli_thread_set_num_threads)(num);
}

pub unsafe fn bli_thread_get_num_threads() -> i32 {
    (get_lib().bli_thread_get_num_threads)()
}

pub unsafe fn omp_get_max_threads() -> c_int {
    (get_lib().omp_get_max_threads)()
}

pub unsafe fn omp_set_num_threads(n: c_int) {
    (get_lib().omp_set_num_threads)(n);
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

fn test_outer_gomp_set() {
    println!("=== Outer, GOMP set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    unsafe { omp_set_num_threads(1) };
    (0..16).into_par_iter().for_each(|i| {
        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { omp_get_max_threads() };
        println!("[Thread] iter {i:2} omp_get_max_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { omp_get_max_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_gomp_set() {
    println!("=== Inner, GOMP set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { omp_set_num_threads(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { omp_get_max_threads() };
        println!("[Thread] iter {i:2} omp_get_max_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { omp_get_max_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_outer_blis_set() {
    println!("=== Outer, BLIS set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    unsafe { bli_thread_set_num_threads(1) };
    (0..16).into_par_iter().for_each(|i| {
        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { bli_thread_get_num_threads() };
        println!("[Thread] iter {i:2} bli_thread_get_num_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { bli_thread_get_num_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn test_inner_blis_set() {
    println!("=== Inner, BLIS set ===");

    let [vec_a, vec_b, vec_c] = gen_vecs();
    (0..16).into_par_iter().for_each(|i| {
        unsafe { bli_thread_set_num_threads(1) };

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        println!("[Thread] iter {i:2} start, rayon thread id: {thread_id:2}");

        let num_threads = unsafe { bli_thread_get_num_threads() };
        println!("[Thread] iter {i:2} bli_thread_get_num_threads: {num_threads}");

        let a = vec_a[i].lock().unwrap();
        let b = vec_b[i].lock().unwrap();
        let mut c = vec_c[i].lock().unwrap();
        run_blas(&a, &b, &mut c);
    });
    let num_threads = unsafe { bli_thread_get_num_threads() };
    println!("[Process] threads after iteration: {num_threads}");
}

fn main() {
    println!("[== BLIS GOMP ==]");
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

    let args = std::env::args().collect::<Vec<_>>();
    assert!(args.len() == 2);
    let mode = &args[1];

    match mode.as_str() {
        "outer-gomp-set" => test_outer_gomp_set(),
        "inner-gomp-set" => test_inner_gomp_set(),
        "outer-blis-set" => test_outer_blis_set(),
        "inner-blis-set" => test_inner_blis_set(),
        _ => panic!("Unknown mode: {mode}"),
    }
}
