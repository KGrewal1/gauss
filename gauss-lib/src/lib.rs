//! If otherwise unstated, equation and page numbers refer to Gaussian Processes for Machine Learning, C. E. Rasmussen & C. K. I. Williams, 2006

#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style
)]
#![allow(clippy::doc_markdown)]

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

/// Trait bounds needed for a type $\text{T}$ to be a valid input for the Gaussian processes
///
/// 1) There must be a covariance function defined on the type with some f64 hyperparameters
///
/// 2) There must be a derivative of this function in terms of the hyperparameters
///
/// Note that the covariance function should never return NaN
pub trait Kernel<const N: usize, Rhs = Self> {
    /// The covariance function $\phi$ on the type $\text{T} --- $
    /// $\phi: (\text{T}, \text{T}, \[\text{f64; \text{N}}\]) \to \text{f64}$
    fn metric(&self, rhs: &Rhs, param: &[f64; N]) -> f64;
    /// The derivative of this covariance function ---
    /// $\phi': (\text{T}, \text{T}, \[\text{f64; \text{N}}\]) \to \[\text{f64; \text{N}}\]$
    fn deriv(&self, rhs: &Rhs, param: &[f64; N]) -> [f64; N];
}

#[derive(Clone, Debug)]

/// Creates a Gaussian Process Solver given a vector of inputs, $x$ and outputs $y$, a value for the variance $\sigma$ and an initial value for the hyperparameters $\theta$
///
/// Given a vector of values of type $\text{T}$ implementing [Kernel]
/// a covariance matric $\bm{K}$ can be constructed
///
/// $\bm{K_{i, j}} = \phi(x_{i}, x_{j}, \theta)+\delta_{i, j}\sigma$
///
/// where $\phi$ is as defined in [Kernel::metric].
///
/// This matrix is positive semi-definite. As such it has a (possibly non-unique) Cholesky (LLT) decomposition.
/// This is decomposition is used to find matrix $X$, the solution to $\bm{K}\bm{X} = \bm{y}$, ie $\bm{K}^{-1}\bm{y}$.
///
/// Direct inversion is not used as this is more numerically stable
/// (p19) and slightly faster for most (direct compuation of the used matrix as opposed to computation of the inverse and then matrix multiplication).
///
/// As the covariance ought to be symmetric, this also leads to the property that $K^{-1}$ is also symmetric
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
    ///
    /// The log marginal likelihood of the process is (eq 2.30)
    ///
    /// $\ln p(y | x, \theta) = -\frac{1}{2}(y^{\intercal}\bm{K}^{-1}y + |\bm{K}| -N\ln 2\pi)$
    ///
    /// where $\phi$ is as defined in [Kernel::metric], $y$ is the vector of outputs, $x$ the vector of inputs, $\theta$ the hyperparamters of $\phi$
    /// and $\bm{K}$ the covariance matrix
    ///
    /// # Errors
    ///
    /// Error if the matrix is non Cholesky decomposable
    pub fn log_marginal_likelihood(&self) -> Result<f64, ProcessError> {
        let x1 = &self.inputs;
        let y1 = &self.res;
        let n = y1.len();
        let autocorr = Mat::from_fn(n, n, |i, j| Kernel::metric(&x1[i], &x1[j], &self.p))
            + Mat::from_fn(n, n, |i, j| if i == j { self.var } else { 0. });
        let y1 = Mat::from_fn(y1.len(), 1, |i, _| y1[i]);
        let Ok(chol_decomp) = autocorr.cholesky(faer::Side::Lower) else {
            return Err(ProcessError::CholeskyFaiure);
        };
        let chol_res = chol_decomp.solve(&y1);
        let yky = (chol_res.transpose() * y1).get(0, 0).to_owned();
        // faster to compute determinant of already decomposed matrix:
        let detk = chol_decomp.compute_l().determinant();
        let ln2pi = (2. * PI).ln();

        // casting the integer size of matrix to float may lose precision
        // however, does not occur for any matrix big enough to be reasonably invertible
        #[allow(clippy::cast_precision_loss)]
        Ok(-0.5 * (yky + detk + (n as f64) * ln2pi))
    }

    /// Calculate the gradient of [GaussProcs::log_marginal_likelihood] with respect to the hyperparameters
    ///
    /// $\frac{\partial }{\partial \theta_{i}}\ln p(y | x, \theta) = \frac{1}{2}\text{Tr}(\bm{K}^{-1}yy^{\intercal}\bm{K}^{-1}\frac{\partial \bm{K}}{\partial \theta_{i}} + \bm{K}^{-1}\frac{\partial \bm{K}}{\partial \theta_{i}})$
    ///
    /// where the components opf the derivative matrix are calculated as
    ///
    /// $\phi_{\theta_{i}}^{'}(x_{i}, x_{j}, \theta)$
    ///
    /// where $\phi'$ is obtained from [Kernel::deriv]
    ///
    /// TODO
    ///
    /// 1) investigate different groupings of the Cholesky solve and / or try explicit inversion for speed
    ///
    /// # Errors
    ///
    /// Error if the matrix is non Cholesky decomposable
    ///
    /// # Panics
    ///
    /// Relies on a Vector with $N$ elements being cast into an array with $N$ :  should always hold
    pub fn gradient(&self) -> Result<[f64; N], ProcessError> {
        let x1 = &self.inputs;
        let y1 = &self.res;
        let n = x1.len();

        let autocorr = Mat::from_fn(n, n, |i, j| Kernel::metric(&x1[i], &x1[j], &self.p))
            + Mat::from_fn(n, n, |i, j| if i == j { self.var } else { 0. });
        println!("autocorr {:?}", autocorr[(22, 36)]);

        let y1 = Mat::from_fn(n, 1, |i, _| y1[i]);
        let Ok(chol_decomp) = autocorr.cholesky(faer::Side::Lower) else {
            return Err(ProcessError::CholeskyFaiure);
        };

        let chol_res = chol_decomp.solve(&y1);

        let mut deriv_mats = vec![Mat::<f64>::zeros(n, n); N];
        for (i, x_1) in x1.iter().enumerate() {
            for (j, x_2) in x1.iter().enumerate() {
                let derivs = Kernel::deriv(x_1, x_2, &self.p);
                for (n, d) in derivs.iter().enumerate() {
                    deriv_mats[n][(i, j)] = *d;
                }
            }
        }
        let dautocorrdps: [Mat<f64>; N] = deriv_mats.try_into().unwrap();
        println!("dautocorrdp {:?}", dautocorrdps[0][(22, 36)]);

        let deltas =
            dautocorrdps.map(|i| (&chol_res * chol_res.transpose()) * &i - chol_decomp.solve(i));

        Ok(deltas.map(|delta| {
            let range = delta.ncols();
            ((0..range).map(|i| delta.get(i, i)).sum::<f64>()) / 2.
        }))
    }

    /// Interpolate the process to a series of points
    ///
    /// Given series of points to predict $z$, return the prediction $\mu$ and its error
    ///
    /// Let there be a cross matrix $\bm{C}$ representing the covariance between points in the original
    /// sample and points in $z$
    ///
    /// $\bm{C_{i, j}} = \phi(x_{i}, z_{j}, \theta)$
    ///
    /// and a posterior matrix $\bm{P}$ representing covariance between points in $z$
    ///
    /// $\bm{P_{i, j}} = \phi(z_{i}, z_{j}, \theta)$
    ///
    /// then the expected value of the points is
    ///
    /// $\mu = \bm{C}^{\intercal}\bm{K}^{-1} y$
    ///
    /// $\mathbb{V}(\mu) = \bm{P} - \bm{C}^{\intercal}\bm{K}^{-1}\bm{C}$
    ///
    /// Eq's 2.25 and 2.26. (Note that $\bm{K}$ has the noise rolled into it)
    ///
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

    /// Interpolate the process to a point, using only $n$ nearest points for this (not hardcoded)
    /// As such significantly reduce the time needed to invert as restrict the size of matric
    ///
    /// Additionally seems to have futher speed up solely through sorting (2x) (ie. when $n$ is the number of points)
    /// possibly due to easier decomposition by structuring matrix
    ///
    ///
    /// # Errors
    /// Error if the matrix is non Cholesky decomposable
    ///
    /// # Panics
    /// Panics if the covariance function returns NaN
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

    /// Equivalent to [GaussProcs::dyn_smart_interpolate] but where $n$ is equal to all elements
    ///
    /// Shows that there is speed up via sortin
    ///
    /// # Errors
    /// Error if the matrix is non Cholesky decomposable
    ///
    /// # Panics
    /// Panics if the metric returns NaN
    #[deprecated]
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

    /// Equivalent to [GaussProcs::dyn_smart_interpolate] but where $n$ is hardcoded
    ///
    /// Existed to if static $n$ allowed for noticeable compiler speed up (did not occur)
    ///
    /// # Errors
    /// Error if the matrix is non Cholesky decomposable
    /// # Panics
    /// Panics if the metric returns NaN
    #[deprecated]
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
