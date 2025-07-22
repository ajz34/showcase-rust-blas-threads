fn main() {
    println!("cargo:rustc-link-search=native=/home/a/Software/OpenBLAS-0.3.28/lib");
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=gomp");
}
