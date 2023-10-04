#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style
)]
use std::{cmp::Reverse, collections::BinaryHeap, f64::consts::PI, fmt::Debug};

use faer::{solvers::Solver, Faer, Mat};
// use nalgebra::{Dyn, Matrix, VecStorage};
use ordered_float::NotNan;
// use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
/// Error in the GP
pub enum ProcessError {
    /// the input arrays are of different length
    MismatchedInputs,
    /// cholesky decomposition failure
    CholeskyFaiure,
}

pub trait Kernel<const N: usize, Rhs = Self> {
    // there must be a 'metric' providing similarity between points
    fn metric(&self, rhs: &Rhs, param: &[f64; N]) -> f64;
    // this must have some derivative
    fn deriv(&self, rhs: &Rhs, param: &[f64; N]) -> [f64; N];
}

#[derive(Clone, Debug)]
pub struct GaussProcs<const N: usize, T>
where
    T: Kernel<N>,
{
    inputs: Vec<T>,
    res: Vec<f64>,
    var: f64,
    p: [f64; N],
}

impl<const N: usize, T> GaussProcs<N, T>
where
    T: Kernel<N>,
{
    /// Creates a new Gaussian Process
    /// # Errors
    /// Returns an error if the number of inputs and the number of outputs provided are non equal
    pub fn new(inputs: Vec<T>, res: Vec<f64>, var: f64, p: [f64; N]) -> Result<Self, ProcessError> {
        if inputs.len() == res.len() {
            Ok(GaussProcs {
                inputs,
                res,
                var,
                p,
            })
        } else {
            Err(ProcessError::MismatchedInputs)
        }
    }

    /// Calculate the log marginal likelihood
    /// # Errors
    /// Error if the matrix is non Cholesky decomposable
    pub fn log_marginal_likelihood(&self) -> Result<f64, ProcessError> {
        let x1 = &self.inputs;
        let y1 = &self.res;
        let n = y1.len();
        let autocorr = Mat::from_fn(n, n, |i, j| Kernel::metric(&x1[i], &x1[j], &self.p))
            + Mat::from_fn(n, n, |i, j| if i == j { self.var } else { 0. });
        let y1 = Mat::from_fn(y1.len(), 1, |i, _| y1[i]);
        let chol_res = match autocorr.cholesky(faer::Side::Lower) {
            Ok(value) => value.solve(&y1),
            Err(_) => return Err(ProcessError::CholeskyFaiure),
        };
        let yky = (chol_res.transpose() * y1).get(0, 0).to_owned();
        let detk = autocorr.determinant();
        let ln2pi = (2. * PI).ln();

        // casting the integer size of matrix to float may lose precision
        // however, does not occur for any matrix big enough to be reasonably invertible
        #[allow(clippy::cast_precision_loss)]
        Ok(-0.5 * (yky + detk + (n as f64) * ln2pi))
    }

    /// Interpolate the process to a series of points
    /// # Errors
    /// Error if the matrix is non Cholesky decomposable
    pub fn interpolate(&self, x2: &[T]) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
        let x1 = &self.inputs;
        let y1 = &self.res;
        let n = x1.len();

        let autocorr = Mat::from_fn(n, n, |i, j| Kernel::metric(&x1[i], &x1[j], &self.p))
            + Mat::from_fn(n, n, |i, j| if i == j { self.var } else { 0. });
        let crosscorr = Mat::from_fn(x1.len(), x2.len(), |i, j| {
            Kernel::metric(&x1[i], &x2[j], &self.p)
        });
        let postcorr = Mat::from_fn(x2.len(), x2.len(), |i, j| {
            Kernel::metric(&x2[i], &x2[j], &self.p)
        });
        let y1 = Mat::from_fn(y1.len(), 1, |i, _| y1[i]);
        // println!("{:?}", autocorr);
        let chol_res = match autocorr.cholesky(faer::Side::Lower) {
            Ok(value) => value.solve(&crosscorr),
            Err(_) => return Err(ProcessError::CholeskyFaiure),
        };

        let mu = { chol_res.transpose() * &y1 };
        let sigma = { postcorr - chol_res.transpose() * crosscorr };
        Ok((mu, sigma))
    }

    /// Interpolate the process to a point
    /// # Errors
    /// Error if the matrix is non Cholesky decomposable
    /// # Panics
    /// Panics if the metric returns NaN
    pub fn interpolate_one(&self, x2: &T) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
        let n = &self.inputs.len();
        let mut heap = BinaryHeap::with_capacity(n + 1);

        self.inputs
            .iter()
            .map(|val| NotNan::new(Kernel::metric(val, x2, &self.p)).expect("NaN from metric"))
            .enumerate()
            .for_each(|(i, num)| {
                heap.push(Reverse((num, i)));
            });

        let indices: Vec<usize> = heap.into_vec().iter().map(|&Reverse((_, i))| i).collect();
        let x1: Vec<&T> = indices.iter().map(|i| &self.inputs[*i]).collect();
        let y1: Vec<&f64> = indices.iter().map(|i| &self.res[*i]).collect();
        let n = x1.len();
        let autocorr = Mat::from_fn(n, n, |i, j| Kernel::metric(x1[i], x1[j], &self.p))
            + Mat::from_fn(n, n, |i, j| if i == j { self.var } else { 0. });
        let crosscorr = Mat::from_fn(n, 1, |i, _| Kernel::metric(x1[i], x2, &self.p));
        let postcorr = Mat::from_fn(1, 1, |_, _| Kernel::metric(x2, x2, &self.p));
        let y1 = Mat::from_fn(n, 1, |i, _| *y1[i]);
        // println!("{:?}", autocorr);
        let chol_res = match autocorr.cholesky(faer::Side::Lower) {
            Ok(value) => value.solve(&crosscorr),
            Err(_) => return Err(ProcessError::CholeskyFaiure),
        };

        let mu = { chol_res.transpose() * &y1 };
        let sigma = { postcorr - chol_res.transpose() * crosscorr };
        Ok((mu, sigma))
    }

    /// Interpolate the process to a point, using only n nearest points for this (hardcoded)
    /// # Errors
    /// Error if the matrix is non Cholesky decomposable
    /// # Panics
    /// Panics if the metric returns NaN
    pub fn smart_interpolate(&self, x2: &T) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
        let n = 6561;

        let mut heap = BinaryHeap::with_capacity(n + 1);

        self.inputs
            .iter()
            .map(|val| NotNan::new(Kernel::metric(val, x2, &self.p)).expect("NaN from metric"))
            .enumerate()
            .for_each(|(i, num)| {
                heap.push(Reverse((num, i)));
                if heap.len() > n {
                    heap.pop();
                }
            });

        let indices: Vec<usize> = heap.into_vec().iter().map(|&Reverse((_, i))| i).collect();

        println!("sorted");
        let x1: Vec<&T> = indices.iter().map(|i| &self.inputs[*i]).collect();
        let y1: Vec<&f64> = indices.iter().map(|i| &self.res[*i]).collect();

        let autocorr = Mat::from_fn(n, n, |i, j| Kernel::metric(x1[i], x1[j], &self.p))
            + Mat::from_fn(n, n, |i, j| if i == j { self.var } else { 0. });
        let crosscorr = Mat::from_fn(n, 1, |i, _| Kernel::metric(x1[i], x2, &self.p));
        let postcorr = Mat::from_fn(1, 1, |_, _| Kernel::metric(x2, x2, &self.p));
        let y1 = Mat::from_fn(n, 1, |i, _| *y1[i]);

        let chol_res = match autocorr.cholesky(faer::Side::Lower) {
            Ok(value) => value.solve(&crosscorr),
            Err(_) => return Err(ProcessError::CholeskyFaiure),
        };

        let mu = { chol_res.transpose() * &y1 };
        let sigma = { postcorr - chol_res.transpose() * crosscorr };
        Ok((mu, sigma))
    }

    /// Interpolate the process to a point, using only n nearest points for this (not hardcoded)
    /// # Errors
    /// Error if the matrix is non Cholesky decomposable
    /// # Panics
    /// Panics if the metric returns NaN
    pub fn dyn_smart_interpolate(
        &self,
        x2: &T,
        n: usize,
    ) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
        // let n = 6561;

        let mut heap = BinaryHeap::with_capacity(n + 1);

        self.inputs
            .iter()
            .map(|val| NotNan::new(Kernel::metric(val, x2, &self.p)).expect("NaN from metric"))
            .enumerate()
            .for_each(|(i, num)| {
                heap.push(Reverse((num, i)));
                if heap.len() > n {
                    heap.pop();
                }
            });

        let indices: Vec<usize> = heap.into_vec().iter().map(|&Reverse((_, i))| i).collect();

        println!("sorted");
        let x1: Vec<&T> = indices.iter().map(|i| &self.inputs[*i]).collect();
        let y1: Vec<&f64> = indices.iter().map(|i| &self.res[*i]).collect();

        let autocorr = Mat::from_fn(n, n, |i, j| Kernel::metric(x1[i], x1[j], &self.p))
            + Mat::from_fn(n, n, |i, j| if i == j { self.var } else { 0. });
        let crosscorr = Mat::from_fn(n, 1, |i, _| Kernel::metric(x1[i], x2, &self.p));
        let postcorr = Mat::from_fn(1, 1, |_, _| Kernel::metric(x2, x2, &self.p));
        let y1 = Mat::from_fn(n, 1, |i, _| *y1[i]);

        let chol_res = match autocorr.cholesky(faer::Side::Lower) {
            Ok(value) => value.solve(&crosscorr),
            Err(_) => return Err(ProcessError::CholeskyFaiure),
        };

        let mu = { chol_res.transpose() * &y1 };
        let sigma = { postcorr - chol_res.transpose() * crosscorr };
        Ok((mu, sigma))
    }
}

// #[cfg(test)]
// mod tests {
//     use std::{
//         cmp::{Ordering, Reverse},
//         collections::BinaryHeap,
//     };

//     use assert_approx_eq::*;
//     use faer::{mat, solvers::Solver, Faer, Mat};
//     use ordered_float::NotNan;
//     // use super::*;
//     // use serde_json::Result;
//     #[test]
//     fn it_works() {
//         let m1 = Mat::from_fn(3, 1, |i, j| i as f64 + j as f64);
//         let a = mat![[7., 2., 1.], [0., 3., -1.], [3., 4., 2f64],];
//         let decomp = a.partial_piv_lu();
//         println!("{}", a.determinant());
//         println!("{:?}", decomp.row_permutation());
//         println!("{:?}", decomp.compute_l());
//         println!("{:?}", decomp.compute_u());
//         // let decomp = a.cholesky(faer::Side::Lower).unwrap();
//         // let sol = decomp.solve(&m1);
//         // let round = a * sol;
//         // for i in 0..3 {
//         //     assert_approx_eq!(m1.get(i, 0), round.get(i, 0))
//         // }
//         assert!(1 == 2);
//         // let mut list = vec![
//         //     NotNan::new(5.).unwrap(),
//         //     NotNan::new(6.).unwrap(),
//         //     NotNan::new(2.).unwrap(),
//         //     NotNan::new(511.).unwrap(),
//         //     NotNan::new(23.).unwrap(),
//         //     NotNan::new(1.).unwrap(),
//         //     NotNan::new(8.).unwrap(),
//         // ];
//         // // list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
//         // // list.iter();
//         // // println!("{:?}", list);
//         // // panic!()
//         // let n = 3; // Replace 3 with any number you want
//         // let mut heap = BinaryHeap::new();

//         // list.iter().enumerate().for_each(|(i, &num)| {
//         //     heap.push(Reverse((num, i)));
//         //     if heap.len() > n {
//         //         heap.pop();
//         //     }
//         // });

//         // let mut largest = heap.into_vec();
//         // largest.sort();

//         // let indices_of_largest: Vec<usize> = largest.iter().map(|&Reverse((_, i))| i).collect();
//         // println!("{:?}", indices_of_largest);
//         panic!()
//     }
// }
