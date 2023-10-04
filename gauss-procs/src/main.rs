use faer::{solvers::Solver, Faer, Mat};
use gauss_lib::*;
use itertools::Itertools;
use std::{f64::consts::PI, time::Instant};
// test function from sfu test functions
// lim et al nonpolynomial function
fn lim_nonpoly(x: &(f64, f64)) -> f64 {
    ((30. + 5. * x.0 * (5. * x.0).sin()) * (4. + (-5. * x.1).exp()) - 100.) / 6.
}

///
/// Gradient of LML, for given metric function f(x1, x2, p)
/// with input variables x, and outputs y is
/// is Tr(y^(T)K-1^(T)K-1y - K-1 dK/dp)/2
/// where K-1 is the inverse of the autocorrelation matrix
fn gradient(p: f64, x1: &[(f64, f64)], y1: &[f64]) -> f64 {
    let n = y1.len();
    let metric = |x: &(f64, f64), y: &(f64, f64)| -> f64 {
        let z2 = p * (((x.0 - y.0).powi(2)) + ((x.1 - y.1).powi(2)));
        (-0.5 * z2).exp()
    };
    let metric_deriv = |x: &(f64, f64), y: &(f64, f64)| -> f64 {
        let z2 = p * (((x.0 - y.0).powi(2)) + ((x.1 - y.1).powi(2)));
        let dz2dp = (((x.0 - y.0).powi(2)) + ((x.1 - y.1).powi(2)));
        -0.5 * dz2dp * (-0.5 * z2).exp()
    };
    let autocorr = Mat::from_fn(n, n, |i, j| (metric)(&x1[i], &x1[j]));
    let dautocorrdp = Mat::from_fn(n, n, |i, j| (metric_deriv)(&x1[i], &x1[j]));
    let y1 = Mat::from_fn(n, 1, |i, _| y1[i]);
    let chol_decomp = autocorr.cholesky(faer::Side::Lower).unwrap();
    let chol_res = chol_decomp.solve(&y1);
    let aadk = (&chol_res * &chol_res.transpose()) * &dautocorrdp;
    let kinvdk = chol_decomp.solve(&dautocorrdp);
    let delta = (aadk - kinvdk);
    let range = delta.ncols();
    let trace = (0..range).map(|i| delta.get(i, i)).sum::<f64>();
    trace / 2.
}

fn lml(p: f64, x1: &[(f64, f64)], y1: &[f64]) -> f64 {
    let n = y1.len();
    let metric = |x: &(f64, f64), y: &(f64, f64)| -> f64 {
        let z2 = p * (((x.0 - y.0).powi(2)) + ((x.1 - y.1).powi(2)));
        (-0.5 * z2).exp()
    };

    let autocorr = Mat::from_fn(n, n, |i, j| (metric)(&x1[i], &x1[j]));
    let y1 = Mat::from_fn(n, 1, |i, _| y1[i]);
    let chol_res = autocorr.cholesky(faer::Side::Lower).unwrap().solve(&y1);
    let yky = (chol_res.transpose() * y1).get(0, 0).to_owned();
    let detk = autocorr.determinant();
    let ln2pi = (2. * PI).ln();
    -0.5 * (yky + detk + (n as f64) * ln2pi)
}

fn metric(x: &(f64, f64), y: &(f64, f64)) -> f64 {
    let z2 = 1750. * (((x.0 - y.0).powi(2)) + ((x.1 - y.1).powi(2)));
    (-0.5 * z2).exp()
}

fn main() {
    // println!("{}", lim_nonpoly((0., 0.)));
    let n: usize = 80;
    let range: Vec<f64> = (0..(n + 1)).map(|i| i as f64 / (n as f64)).collect();
    let inputs: Vec<(f64, f64)> = range.clone().into_iter().cartesian_product(range).collect();
    let outputs: Vec<f64> = inputs.iter().map(|x| lim_nonpoly(x)).collect();
    let v = 1500.;
    println!("{}", gradient(v, &inputs, &outputs));
    println!("{}", lml(v, &inputs, &outputs));
    // println!("{:?}", inputs);
    // println!("{:?}", outputs);
    // println!("{:?}, {:?}", &inputs[100], &inputs[99]);
    // println!("{}", metric(&inputs[100], &inputs[99]));
    // println!("{}", inputs.len());
    // let proc = GaussProcs::new(inputs, outputs, 0., metric).unwrap();
    // println!("{}", proc.log_marginal_likelihood().unwrap());

    // let now = Instant::now();
    // println!("{:?}", proc.smart_interpolate(&(0.215, 0.255)).unwrap());
    // let elapsed = now.elapsed();
    // println!("Elapsed, trunc: {:.2?}", elapsed);

    // let now = Instant::now();
    // println!(
    //     "{:?}",
    //     proc.dyn_smart_interpolate(&(0.215, 0.255), 6561).unwrap()
    // );
    // let elapsed = now.elapsed();
    // println!("Elapsed, trunc: {:.2?}", elapsed);

    // let now = Instant::now();
    // println!("{:?}", proc.interpolate(&[(0.215, 0.255)]).unwrap());
    // let elapsed = now.elapsed();
    // println!("Elapsed, non trunc: {:.2?}", elapsed);

    // let now = Instant::now();

    // println!("{:?}", proc.interpolate_one(&(0.215, 0.255)).unwrap());
    // let elapsed = now.elapsed();
    // println!("Elapsed, trunc: {:.2?}", elapsed);

    // let predicted = proc
    //     .interpolate_one(&(0.215, 0.255))
    //     .unwrap()
    //     .0
    //     .get(0, 0)
    //     .clone();

    // println!("{:?}", lim_nonpoly(&(0.215, 0.255)));
    // println!("{}", lim_nonpoly(&(0.215, 0.255)) - predicted)
}
