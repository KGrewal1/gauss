use std::f64::consts::PI;

use gauss_lib::*;
use itertools::Itertools;
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use statrs::function::erf::erf;
/// Rosenbrock function to test optimisation
/// global minimum should be at (1,1) with value -1
fn rosenbrock(x: &TwoDpoint) -> f64 {
    (1. - x.0).powi(2) + 10. * (x.1 - x.0.powi(2)).powi(2) - 1.
}

/// standard normal distribution (mean 0, std 1)
fn norm(x: f64) -> f64 {
    ((-0.5 * x.powi(2)).exp()) / ((2. * PI).sqrt())
}
/// Expected improement vs minimum given current min, EV and error
fn expected_improvement(min: f64, mu: f64, sigma: f64, zeta: f64) -> f64 {
    (min - mu - zeta) * erf((min - mu - zeta) / sigma) + sigma * norm(min - mu - zeta)
}

#[derive(Clone, Copy, Debug)]
struct TwoDpoint(f64, f64);

impl Kernel<1> for TwoDpoint {
    fn metric(&self, rhs: &Self, param: &[f64; 1]) -> f64 {
        let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
        (-0.5 * z2).exp()
    }

    fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
        let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
        let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
        [-0.5 * dz2dp * (-0.5 * z2).exp()]
    }
}

fn main() {
    // LHS Sample
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let n: usize = 5; // number of points to sample
    let x_range = (-5., 10.);
    let y_range = (-5., 10.);
    let mut range_1: Vec<usize> = (0..(n + 1)).collect();
    let mut range_2 = range_1.clone();
    range_1.shuffle(&mut rng);
    range_2.shuffle(&mut rng);
    let samples: Vec<TwoDpoint> = range_1
        .into_iter()
        .zip(range_2)
        .map(|(i, j)| {
            let x = (i as f64 + x_range.0) * (x_range.1 - x_range.0) / (n as f64);
            let y = (j as f64 + y_range.0) * (y_range.1 - y_range.0) / (n as f64);
            TwoDpoint(x, y)
        })
        .collect();
    let outputs: Vec<f64> = samples.iter().map(rosenbrock).collect();

    println!("{:?}", outputs);
    // create process
    let mut proc = GaussProcs::new(samples.clone(), outputs.clone(), 0.01, [3.5]).unwrap(); //
    println!("{}", proc.log_marginal_likelihood().unwrap());
    println!("{:?}", proc.gradient().unwrap());

    // generate input to test
    let n: usize = 30;
    let range: Vec<f64> = (0..(n + 1))
        .map(|i| (i as f64 + x_range.0) * (x_range.1 - x_range.0) / (n as f64))
        .collect();
    let test_inputs: Vec<TwoDpoint> = range
        .clone()
        .into_iter()
        .cartesian_product(range)
        .map(|(i, j)| TwoDpoint(i, j))
        .collect();
    // println!("{:?}", &test_inputs[132..150]);
    // panic!();
    let factor = 15.;
    let zeta = 175.;
    let mut minimum = outputs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    println!("Initial minimum {}", minimum);
    for _i in 1..75 {
        let (mu, sigma) = proc.interpolate(&test_inputs).unwrap();
        // println!("{:?}", mu);
        // println!("{:?},{:?}", test_inputs[0], mu.get(0, 0));
        let mut res: Vec<(TwoDpoint, f64, f64, usize, f64)> = test_inputs
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, input)| {
                (
                    input,
                    *mu.get(i, 0),
                    *sigma.get(i, i),
                    i,
                    expected_improvement(minimum, *mu.get(i, 0), *sigma.get(i, i), zeta),
                )
            })
            .collect();
        res.sort_by(|a, b| (b.4).partial_cmp(&(a.4)).unwrap());
        let next = res.pop().unwrap();
        println!("Next {:?}", next);
        let val = rosenbrock(&next.0);
        println!("{}", val);
        if val < minimum {
            println!("Switch");
            minimum = val
        }
        proc.update(next.0, val).unwrap();
        // println!("{:?}", res[224])
    }

    let mut proc = GaussProcs::new(samples.clone(), outputs.clone(), 0.01, [3.5]).unwrap();
    // panic!();
    // this loop effectivley uses UCB to converge on minimum
    for _i in 1..100 {
        let (mu, sigma) = proc.interpolate(&test_inputs).unwrap();
        // println!("{:?}", mu);
        // println!("{:?},{:?}", test_inputs[0], mu.get(0, 0));
        let mut res: Vec<(TwoDpoint, f64, f64, usize)> = test_inputs
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, input)| (input, *mu.get(i, 0), *sigma.get(i, i), i))
            .collect();
        res.sort_by(|a, b| {
            (b.1 - factor * b.2)
                .partial_cmp(&(a.1 - factor * a.2))
                .unwrap()
        });
        let next = res.pop().unwrap();
        println!("Next {:?}", next);
        let val = rosenbrock(&next.0);
        proc.update(next.0, val).unwrap();
    }
}
