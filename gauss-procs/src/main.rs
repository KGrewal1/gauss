use gauss_lib::*;
use itertools::Itertools;
use std::time::Instant;
// test function from sfu test functions
// lim et al nonpolynomial function
fn lim_nonpoly(x: &(f64, f64)) -> f64 {
    ((30. + 5. * x.0 * (5. * x.0).sin()) * (4. + (-5. * x.1).exp()) - 100.) / 6.
}

fn metric(x: &(f64, f64), y: &(f64, f64)) -> f64 {
    let z2 = 5_000. * (((x.0 - y.0).powi(2)) + ((x.1 - y.1).powi(2)));
    (-0.5 * z2).exp()
}

fn main() {
    // println!("{}", lim_nonpoly((0., 0.)));
    let n: usize = 80;
    let range: Vec<f64> = (0..(n + 1)).map(|i| i as f64 / (n as f64)).collect();
    let inputs: Vec<(f64, f64)> = range.clone().into_iter().cartesian_product(range).collect();
    let outputs: Vec<f64> = inputs.iter().map(|x| lim_nonpoly(x)).collect();
    // println!("{:?}", inputs);
    // println!("{:?}", outputs);
    println!("{:?}, {:?}", &inputs[100], &inputs[99]);
    println!("{}", metric(&inputs[100], &inputs[99]));
    println!("{}", inputs.len());
    let proc = GaussProcs::new(inputs, outputs, 0., metric).unwrap();

    let now = Instant::now();
    println!("{:?}", proc.smart_interpolate(&(0.215, 0.255)).unwrap());
    let elapsed = now.elapsed();
    println!("Elapsed, trunc: {:.2?}", elapsed);

    let now = Instant::now();
    println!(
        "{:?}",
        proc.dyn_smart_interpolate(&(0.215, 0.255), 6561).unwrap()
    );
    let elapsed = now.elapsed();
    println!("Elapsed, trunc: {:.2?}", elapsed);

    let now = Instant::now();
    println!("{:?}", proc.interpolate(&[(0.215, 0.255)]).unwrap());
    let elapsed = now.elapsed();
    println!("Elapsed, non trunc: {:.2?}", elapsed);

    let now = Instant::now();
    println!("{:?}", proc.interpolate_one(&(0.215, 0.255)).unwrap());
    let elapsed = now.elapsed();
    println!("Elapsed, trunc: {:.2?}", elapsed);

    println!("{:?}", lim_nonpoly(&(0.215, 0.255)));
}
