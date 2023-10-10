//! This library implements Gaussian processes (GP) for use in Bayesian Regression
//! with an aim of use in minimisation of chemical structures.
//!
//! GP regression treats the unknown function values at all training points as a
//! Gaussian Process fully characterised by some arbitrary covariance function or
//! kernel between points $x_{1}$ and $x_{2}$.
//!
//! The output is modelled as a stochastic process:
//!
//! $ Y(x) = \mu(x) + Z(x)$
//!
//! where $\mu$ is the mean and $Z(x)$ realization of the stochastic process
//!
//! # Implementation
//!
//! * Based of [Faer](https://github.com/sarah-ek/faer-rs) to provide linear algebra subroutine
//! * GP mean currently must be 0
//! * GP correlation function can be arbitrary, with parameterisation
//! * Dimensionality issue : scales as $O(n^{3})$ in size of training data
//! * 'Focused' regression, utilising only training points most similar to test point to reduce complexity
//!
//!
//! # Reference
//!
//! Gaussian Processes for Machine Learning, C. E. Rasmussen & C. K. I. Williams, 2006
//!
//! Jones, M. R., et al. "Constraining Gaussian processes for grey-box acoustic emission source localisation."
//! Proceedings of the 29th international conference on noise and vibration engineering (ISMA 2020). 2020.
//!
//! Jan N. Fuhg, Michele Marino, Nikolaos Bouklas,
//! Local approximate Gaussian process regression for data-driven constitutive models: development and comparison with neural networks,
//! Computer Methods in Applied Mechanics and Engineering,
//! Volume 388,
//! 2022,
//! 114217,
//! ISSN 0045-7825,
//! <https://doi.org/10.1016/j.cma.2021.114217>.
//!
//! arXiv:1402.0645 [cs.LG]
//! <https://doi.org/10.48550/arXiv.1402.0645>
//!
//! Lafage, R., (2022). egobox, a Rust toolbox for efficient global optimization.
//! Journal of Open Source Software, 7(78), 4737,
//! <https://doi.org/10.21105/joss.04737>
//!
//! arXiv:2306.06475 [cond-mat.mtrl-sci]
//! <https://doi.org/10.48550/arXiv.2306.06475>

#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style
)]
#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown)]

use std::{cmp::Reverse, collections::BinaryHeap, f64::consts::PI, fmt::Debug};

use dyn_stack::{GlobalPodBuffer, PodStack};
use faer::{solvers::Solver, Faer, Mat};
use faer_cholesky::llt::{
    update::{insert_rows_and_cols_clobber, insert_rows_and_cols_clobber_req},
    CholeskyError,
};
use faer_core::Parallelism;
// use nalgebra::{Dyn, Matrix, VecStorage};
use ordered_float::NotNan;
// use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
/// Error in the GP
pub enum ProcessError {
    /// the input arrays are of different length
    MismatchedInputs,
    /// cholesky decomposition failure
    CholeskyFaiure,
}

impl From<CholeskyError> for ProcessError {
    fn from(_e: CholeskyError) -> ProcessError {
        ProcessError::CholeskyFaiure
    }
}

/// find x, the solution to AX=B where a is posiive definite
/// takes cholesky decomposition of A as an input
fn cholesky_solve(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let mut b = b.clone();
    let i = a.nrows();
    let j = b.ncols();

    faer_cholesky::llt::solve::solve_in_place_with_conj(
        a.as_ref(),
        faer_core::Conj::No,
        b.as_mut(),
        Parallelism::Rayon(0),
        PodStack::new(&mut GlobalPodBuffer::new(
            faer_cholesky::llt::solve::solve_in_place_req::<f64>(j, i, Parallelism::Rayon(0))
                .unwrap(),
        )),
    );

    b
}

/// Trait bounds needed for a type $\text{T}$ to be a valid input for the Gaussian processes
///
/// 1. There must be a covariance function defined on the type with some f64 hyperparameters
///
/// 2. There must be a derivative of this function in terms of the hyperparameters
///
/// Note that the covariance function should never return NaN
///
/// # Examples
///
/// Defining the RBF Kernel on a 2-D point
/// ```
/// use gauss_lib::Kernel;
///
/// pub struct TwoDpoint(f64, f64);
///
/// impl Kernel<1> for TwoDpoint {
///     fn metric(&self, rhs: &Self, param: &[f64; 1]) -> f64 {
///         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
///         (-0.5 * z2).exp()
///     }
///     fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
///         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
///         let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
///         [-0.5 * dz2dp * (-0.5 * z2).exp()]
///     }
/// }
/// ```
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
    autocorr: Mat<f64>,
    cholesky_l: Mat<f64>,
}

impl<const N: usize, T> GaussProcs<N, T>
where
    T: Kernel<N>,
{
    /// Calculate the autocorrelation matrix
    fn autocorr(&self, x1: &[&T]) -> Mat<f64> {
        let n = x1.len();
        Mat::from_fn(n, n, |i, j| Kernel::metric(x1[i], x1[j], &self.p))
            + Mat::from_fn(n, n, |i, j| if i == j { self.var } else { 0. })
    }

    /// Internal wrapper around cholesky solve
    /// to ensure arguments are ordered correctly
    fn cholesky_solve(&self, b: &Mat<f64>) -> Mat<f64> {
        cholesky_solve(&self.cholesky_l, b)
    }

    /// Creates a new Gaussian Process
    ///
    /// # Errors
    ///
    /// Returns an error if the number of inputs and the number of outputs provided are non equal
    ///
    /// Returns an error if the matrix is non Cholesky decomposable
    ///
    /// # Examples
    ///
    /// ```
    /// # use gauss_lib::*;
    /// #
    /// pub struct TwoDpoint(f64, f64);
    /// #
    /// # impl Kernel<1> for TwoDpoint {
    /// #     fn metric(&self, rhs: &Self, param: &[f64; 1]) -> f64 {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         (-0.5 * z2).exp()
    /// #     }
    /// #     fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
    /// #         [-0.5 * dz2dp * (-0.5 * z2).exp()]
    /// #     }
    /// # }
    ///
    /// let points = vec![TwoDpoint(1., 1.), TwoDpoint(1.,2.)];
    /// let vals = vec![1., 2.];
    /// let proc = GaussProcs::new(points, vals, 0., [10.]).unwrap();
    /// ```
    pub fn new(inputs: Vec<T>, res: Vec<f64>, var: f64, p: [f64; N]) -> Result<Self, ProcessError> {
        if inputs.len() == res.len() {
            let n = res.len();
            let autocorr = Mat::from_fn(n, n, |i, j| Kernel::metric(&inputs[i], &inputs[j], &p))
                + Mat::from_fn(n, n, |i, j| if i == j { var } else { 0. });
            let chol_decomp = autocorr.cholesky(faer::Side::Lower)?;
            let cholesky_l = chol_decomp.compute_l();
            Ok(GaussProcs {
                inputs,
                res,
                var,
                p,
                autocorr,
                cholesky_l,
            })
        } else {
            Err(ProcessError::MismatchedInputs)
        }
    }

    /// Updates a new Gaussian Process
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix stops being Cholesky invertible
    ///
    /// # Panics
    /// Panics if OOM on allocating additional memory for extending the decomposition
    ///
    /// # Examples
    ///
    /// ```
    /// # use gauss_lib::*;
    /// #
    /// pub struct TwoDpoint(f64, f64);
    /// #
    /// # impl Kernel<1> for TwoDpoint {
    /// #     fn metric(&self, rhs: &Self, param: &[f64; 1]) -> f64 {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         (-0.5 * z2).exp()
    /// #     }
    /// #     fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
    /// #         [-0.5 * dz2dp * (-0.5 * z2).exp()]
    /// #     }
    /// # }
    ///
    /// let points = vec![TwoDpoint(1., 1.), TwoDpoint(1.,2.)];
    /// let vals = vec![1., 2.];
    /// let mut proc = GaussProcs::new(points, vals, 0., [10.]).unwrap();
    /// let new_point = TwoDpoint(1., 3.);
    /// let new_val = 3.;
    /// proc.update(new_point, new_val);
    /// ```
    pub fn update(&mut self, input: T, res: f64) -> Result<(), ProcessError> {
        // add the new input and output
        self.inputs.push(input);
        self.res.push(res);

        // edit the autocorrelation matrix
        let n = self.inputs.len();
        self.autocorr.resize_with(n, n, |i, j| {
            Kernel::metric(&self.inputs[i], &self.inputs[j], &self.p)
                + if i == j { self.var } else { 0. }
        });

        // update the cholesky decomposition
        let mut new_col = self.autocorr.get(0..n, n - 1).to_owned();
        self.cholesky_l.resize_with(n, n, |_, _| 0.);
        Ok(insert_rows_and_cols_clobber(
            self.cholesky_l.as_mut(),
            n - 1,
            new_col.as_mut(),
            Parallelism::Rayon(0),
            PodStack::new(&mut GlobalPodBuffer::new(
                insert_rows_and_cols_clobber_req::<f64>(n, Parallelism::Rayon(0)).unwrap(),
            )),
        )?)
    }

    /// Calculate the log marginal likelihood
    ///
    /// The log marginal likelihood of the process is (eq 2.30)
    ///
    /// $\ln p(y | x, \theta) = -\frac{1}{2}(y^{\intercal}\bm{K}^{-1}y + |\bm{K}| +N\ln 2\pi)$
    ///
    /// where $\phi$ is as defined in [Kernel::metric], $y$ is the vector of outputs, $x$ the vector of inputs, $\theta$ the hyperparamters of $\phi$
    /// and $\bm{K}$ the covariance matrix
    ///
    /// # Examples
    ///
    /// ```
    /// # use gauss_lib::*;
    /// #
    /// pub struct TwoDpoint(f64, f64);
    /// #
    /// # impl Kernel<1> for TwoDpoint {
    /// #     fn metric(&self, rhs: &Self, param: &[f64; 1]) -> f64 {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         (-0.5 * z2).exp()
    /// #     }
    /// #     fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
    /// #         [-0.5 * dz2dp * (-0.5 * z2).exp()]
    /// #     }
    /// # }
    ///
    /// let points = vec![TwoDpoint(1., 1.), TwoDpoint(1.,2.)];
    /// let vals = vec![1., 2.];
    /// let proc = GaussProcs::new(points, vals, 0., [10.]).unwrap();
    /// proc.log_marginal_likelihood();
    /// ```
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn log_marginal_likelihood(&self) -> f64 {
        let y1 = &self.res;
        let n = y1.len();
        let y1 = Mat::from_fn(y1.len(), 1, |i, _| y1[i]);

        let chol_res = self.cholesky_solve(&y1);
        let yky = (chol_res.transpose() * y1).get(0, 0).to_owned();
        // faster to compute determinant of already decomposed matrix:
        // the determinant is the square of the determinant of L in LLLT decomp
        let detk = self.cholesky_l.determinant().powi(2);
        let ln2pi = (2. * PI).ln();

        // casting the integer size of matrix to float may lose precision
        // however, does not occur for any matrix big enough to be reasonably invertible
        -0.5 * (yky + detk + (n as f64) * ln2pi)
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
    /// # TODO
    ///
    /// 1) investigate different groupings of the Cholesky solve and / or try explicit inversion for speed
    ///
    /// # Panics
    ///
    /// Relies on a Vector with $N$ elements being cast into an array with $N$ :  should always hold
    ///
    /// # Examples
    ///
    /// ```
    /// # use gauss_lib::*;
    /// #
    /// pub struct TwoDpoint(f64, f64);
    /// #
    /// # impl Kernel<1> for TwoDpoint {
    /// #     fn metric(&self, rhs: &Self, param: &[f64; 1]) -> f64 {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         (-0.5 * z2).exp()
    /// #     }
    /// #     fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
    /// #         [-0.5 * dz2dp * (-0.5 * z2).exp()]
    /// #     }
    /// # }
    ///
    /// let points = vec![TwoDpoint(1., 1.), TwoDpoint(1.,2.)];
    /// let vals = vec![1., 2.];
    /// let proc = GaussProcs::new(points, vals, 0., [10.]).unwrap();
    /// proc.gradient();
    /// ```
    #[must_use]
    pub fn gradient(&self) -> [f64; N] {
        let x1 = &self.inputs.iter().collect::<Vec<&T>>();
        let y1 = &self.res;
        let n = x1.len();

        let y1 = Mat::from_fn(n, 1, |i, _| y1[i]);

        let chol_res = self.cholesky_solve(&y1);

        let mut deriv_mats = vec![Mat::<f64>::zeros(n, n); N];
        for (i, x_1) in x1.iter().enumerate() {
            for (j, x_2) in x1.iter().enumerate() {
                let derivs = Kernel::deriv(*x_1, *x_2, &self.p);
                for (n, d) in derivs.iter().enumerate() {
                    deriv_mats[n][(i, j)] = *d;
                }
            }
        }
        let dautocorrdps: [Mat<f64>; N] = deriv_mats.try_into().unwrap();

        let deltas = dautocorrdps.map(|i| {
            (&chol_res * chol_res.transpose()) * &i - cholesky_solve(&self.cholesky_l, &i)
        });

        deltas.map(|delta| {
            let range = delta.ncols();
            ((0..range).map(|i| delta.get(i, i)).sum::<f64>()) / 2.
        })
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
    /// # Examples
    ///
    /// ```
    /// # use gauss_lib::*;
    /// #
    /// pub struct TwoDpoint(f64, f64);
    /// #
    /// # impl Kernel<1> for TwoDpoint {
    /// #     fn metric(&self, rhs: &Self, param: &[f64; 1]) -> f64 {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         (-0.5 * z2).exp()
    /// #     }
    /// #     fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
    /// #         [-0.5 * dz2dp * (-0.5 * z2).exp()]
    /// #     }
    /// # }
    ///
    /// let points = vec![TwoDpoint(1., 1.), TwoDpoint(1.,2.)];
    /// let vals = vec![1., 2.];
    /// let proc = GaussProcs::new(points, vals, 0., [10.]).unwrap();
    /// proc.log_marginal_likelihood();
    /// let new_point = TwoDpoint(1., 1.5);
    /// proc.interpolate(&[new_point]);
    /// ```
    pub fn interpolate(&self, x2: &[T]) -> (Mat<f64>, Mat<f64>) {
        let x1 = &self.inputs;
        let y1 = &self.res;

        let crosscorr = Mat::from_fn(x1.len(), x2.len(), |i, j| {
            Kernel::metric(&x1[i], &x2[j], &self.p)
        });
        let postcorr = Mat::from_fn(x2.len(), x2.len(), |i, j| {
            Kernel::metric(&x2[i], &x2[j], &self.p)
        });
        let y1 = Mat::from_fn(y1.len(), 1, |i, _| y1[i]);

        let chol_res = self.cholesky_solve(&crosscorr);

        let mu = { chol_res.transpose() * &y1 };
        let sigma = { postcorr - chol_res.transpose() * crosscorr };
        (mu, sigma)
    }

    /// Interpolate the process to a point, using only $n$ nearest points for this (not hardcoded)
    /// As such significantly reduce the time needed to invert as restrict the size of matric
    ///
    /// Additionally seems to have futher speed up solely through sorting (2x) (ie. when $n$ is the number of points)
    /// possibly due to easier decomposition by structuring matrix
    ///
    ///
    /// # Errors
    ///
    /// Error if the matrix is non Cholesky decomposable
    ///
    /// # Panics
    ///
    /// Panics if the covariance function returns NaN
    ///
    /// # Examples
    ///
    /// ```
    /// # use gauss_lib::*;
    /// #
    /// pub struct TwoDpoint(f64, f64);
    /// #
    /// # impl Kernel<1> for TwoDpoint {
    /// #     fn metric(&self, rhs: &Self, param: &[f64; 1]) -> f64 {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         (-0.5 * z2).exp()
    /// #     }
    /// #     fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
    /// #         let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
    /// #         let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
    /// #         [-0.5 * dz2dp * (-0.5 * z2).exp()]
    /// #     }
    /// # }
    ///
    /// let points = vec![TwoDpoint(1., 1.), TwoDpoint(1.,2.)];
    /// let vals = vec![1., 2.];
    /// let proc = GaussProcs::new(points, vals, 0., [10.]).unwrap();
    /// proc.log_marginal_likelihood();
    /// let new_point = TwoDpoint(1., 1.5);
    /// proc.dyn_smart_interpolate(&new_point, 2);
    /// ```
    pub fn dyn_smart_interpolate(
        &self,
        x2: &T,
        n: usize,
    ) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
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

        let n = indices.len();

        let x1: Vec<&T> = indices.iter().map(|i| &self.inputs[*i]).collect();
        let y1: Vec<&f64> = indices.iter().map(|i| &self.res[*i]).collect();

        let autocorr = self.autocorr(&x1);
        let crosscorr = Mat::from_fn(n, 1, |i, _| Kernel::metric(x1[i], x2, &self.p));
        let postcorr = Mat::from_fn(1, 1, |_, _| Kernel::metric(x2, x2, &self.p));
        let y1 = Mat::from_fn(n, 1, |i, _| *y1[i]);

        let chol_res = autocorr.cholesky(faer::Side::Lower)?.solve(&crosscorr);

        let mu = { chol_res.transpose() * &y1 };
        let sigma = { postcorr - chol_res.transpose() * crosscorr };
        Ok((mu, sigma))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use faer::{assert_matrix_eq, mat};
    use itertools::Itertools;

    fn lim_nonpoly(x: &TwoDpoint) -> f64 {
        ((30. + 5. * x.0 * (5. * x.0).sin()) * (4. + (-5. * x.1).exp()) - 100.) / 6.
    }

    fn lim_nonpoly_bad(x: &BadTwoDpoint) -> f64 {
        ((30. + 5. * x.0 * (5. * x.0).sin()) * (4. + (-5. * x.1).exp()) - 100.) / 6.
    }

    #[derive(Debug)]
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

    #[derive(Debug)]
    struct BadTwoDpoint(f64, f64);

    impl Kernel<1> for BadTwoDpoint {
        fn metric(&self, _rhs: &Self, _param: &[f64; 1]) -> f64 {
            1.
        }

        fn deriv(&self, rhs: &Self, param: &[f64; 1]) -> [f64; 1] {
            let z2 = param[0] * (((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2)));
            let dz2dp = ((self.0 - rhs.0).powi(2)) + ((self.1 - rhs.1).powi(2));
            [-0.5 * dz2dp * (-0.5 * z2).exp()]
        }
    }

    #[test]
    fn full_test() {
        let n: usize = 10;
        let range: Vec<f64> = (0..(n + 1)).map(|i| i as f64 / (n as f64)).collect();
        let inputs: Vec<TwoDpoint> = range
            .clone()
            .into_iter()
            .cartesian_product(range)
            .map(|(i, j)| TwoDpoint(i, j))
            .collect();
        let outputs: Vec<f64> = inputs.iter().map(lim_nonpoly).collect();

        let mut proc = GaussProcs::new(inputs, outputs, 0., [1750.]).unwrap();
        let grad = proc.gradient();
        let lml = proc.log_marginal_likelihood();
        println!("{:?}, {}", grad, lml);
        proc.dyn_smart_interpolate(&TwoDpoint(0.215, 0.255), 15)
            .unwrap();
        proc.interpolate(&[TwoDpoint(0.215, 0.255)]);
        let new_point = TwoDpoint(0.215, 0.255);
        let new_res = lim_nonpoly(&new_point);
        proc.update(new_point, new_res).unwrap();
    }

    // mismatched size of inputs results in an error
    #[test]
    fn check_len() {
        let n: usize = 10;
        let range: Vec<f64> = (0..(n + 1)).map(|i| i as f64 / (n as f64)).collect();
        let inputs: Vec<TwoDpoint> = range
            .clone()
            .into_iter()
            .cartesian_product(range)
            .map(|(i, j)| TwoDpoint(i, j))
            .collect();
        let mut outputs = inputs.iter().map(lim_nonpoly).collect::<Vec<f64>>();

        outputs.pop();
        assert_eq!(
            GaussProcs::new(inputs, outputs, 0., [1750.]).unwrap_err(),
            ProcessError::MismatchedInputs
        )
    }

    // inputs causing singular matrices result in an erro
    #[test]
    fn singular_mat() {
        let n: usize = 10;
        let range: Vec<f64> = (0..(n + 1)).map(|i| i as f64 / (n as f64)).collect();
        let inputs: Vec<BadTwoDpoint> = range
            .clone()
            .into_iter()
            .cartesian_product(range)
            .map(|(i, j)| BadTwoDpoint(i, j))
            .collect();
        let outputs = inputs.iter().map(lim_nonpoly_bad).collect::<Vec<f64>>();

        // let proc = GaussProcs::new(inputs, outputs, 0., [1750.]).unwrap();

        assert_eq!(
            GaussProcs::new(inputs, outputs, 0., [1750.]).unwrap_err(),
            ProcessError::CholeskyFaiure
        );
    }

    // update throws error if it results in a singular matrix
    #[test]
    fn update_err() {
        let inputs: Vec<TwoDpoint> = vec![TwoDpoint(1., 1.), TwoDpoint(1., 2.)];
        let outputs = vec![1., 2.];

        let mut proc = GaussProcs::new(inputs, outputs, 0., [1750.]).unwrap();

        let new_point = TwoDpoint(1., 1.);
        let new_val = 3.;

        assert_eq!(
            proc.update(new_point, new_val).unwrap_err(),
            ProcessError::CholeskyFaiure
        );
    }

    #[test]
    fn cholesky_solve_test() {
        // check the decomposition is as expected
        let initial = mat!([4., 12.], [12., 37.]);
        let mut decomp = initial.cholesky(faer::Side::Lower).unwrap().compute_l();
        let expected_res = mat!([2., 0.], [6., 1.]);
        assert_matrix_eq!(decomp, expected_res, comp = float);

        // check the solve works as expected
        let target = mat!([1., 2.], [3., 4.]);
        let res = cholesky_solve(&decomp, &target);
        let expected_res = mat!([0.25, 6.5], [0., -2.]);
        assert_matrix_eq!(res, expected_res, comp = float);

        // check adding a column goes as expected
        decomp.resize_with(3, 3, |_, _| 0.);
        let mut new_col = mat!([-16.], [-43.], [98.]);

        insert_rows_and_cols_clobber(
            decomp.as_mut(),
            2,
            new_col.as_mut(),
            Parallelism::Rayon(0),
            PodStack::new(&mut GlobalPodBuffer::new(
                insert_rows_and_cols_clobber_req::<f64>(3, Parallelism::Rayon(0)).unwrap(),
            )),
        )
        .unwrap();
        let expected_res = mat!([2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]);
        assert_matrix_eq!(decomp, expected_res, comp = float);

        // check the solve works as expected
        let target = mat!([3., 6., 2.], [4., 3., 6.], [0., 1., 3.]);
        let res = cholesky_solve(&decomp, &target);
        println!("{:?}", target);
        let expected_res = mat!(
            [3379. / 36., 4637. / 18., 427. / 18.],
            [-230. / 9., -635. / 9., -55. / 9.],
            [37. / 9., 100. / 9., 11. / 9.]
        );
        assert_matrix_eq!(res, expected_res, comp = float);
    }
}
