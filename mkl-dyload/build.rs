fn main() {
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-lm");
}
