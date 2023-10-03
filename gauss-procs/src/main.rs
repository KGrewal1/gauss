use gauss_lib::*;
fn main() {
    fn f(x: &f64, y: &f64) -> f64 {
        let z = (x - y).abs();
        (-0.5 * z.powi(2)).exp()
    }
    let proc = GaussProcs::new(vec![1., 2.], vec![1., 2.], 0., f).unwrap();
    println!("{:?}", proc.interpolate(&[1.5]));
}
