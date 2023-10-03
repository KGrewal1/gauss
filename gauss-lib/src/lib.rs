#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style
)]
use faer::{solvers::Solver, Faer, IntoFaer, IntoNalgebra, Mat};
use nalgebra::{Dyn, Matrix, VecStorage};
use serde::{Deserialize, Serialize};

#[must_use]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[derive(Debug, Clone)]
pub enum ProcessError {
    MismatchedInputs,
    CholeskyFaiure,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Internal component for serialisation and deserialisation
struct Process<T> {
    inputs: Vec<T>, // the list of input values with know outputs
    res: Vec<f64>,  // list of the known outputs
    var: f64,       // the variance
    autocorr: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>, // autocorrelation matrix
}

#[derive(Clone, Debug)]
pub struct GaussProcs<T> {
    process: Process<T>,
    pub metric: fn(&T, &T) -> f64,
}

impl<T> GaussProcs<T> {
    pub fn new(
        vals: Vec<T>,
        res: Vec<f64>,
        var: f64,
        metric: fn(&T, &T) -> f64,
    ) -> Result<Self, ProcessError> {
        if vals.len() == res.len() {
            let m2 = Mat::from_fn(vals.len(), vals.len(), |i, j| metric(&vals[i], &vals[j]))
                + Mat::from_fn(
                    vals.len(),
                    vals.len(),
                    |i, j| if i == j { var } else { 0.0 },
                );
            let m = m2.as_ref().into_nalgebra();
            let process = Process {
                inputs: vals,
                res,
                var,
                autocorr: m.into(),
            };
            Ok(GaussProcs { process, metric })
        } else {
            Err(ProcessError::MismatchedInputs)
        }
    }

    pub fn interpolate(&self, x2: &[T]) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
        let x1 = &self.process.inputs;
        let y1 = &self.process.res;
        let autocorr: Mat<f64> = self
            .process
            .autocorr
            .view_range(.., ..)
            .into_faer()
            .to_owned();
        let crosscorr = Mat::from_fn(x1.len(), x2.len(), |i, j| (self.metric)(&x1[i], &x2[j]));
        let postcorr = Mat::from_fn(x2.len(), x2.len(), |i, j| (self.metric)(&x2[i], &x2[j]));
        let y1 = Mat::from_fn(y1.len(), 1, |i, _| y1[i]);
        let chol_res = match autocorr.cholesky(faer::Side::Lower) {
            Ok(value) => value.solve(&y1),
            Err(_) => return Err(ProcessError::CholeskyFaiure),
        };

        let mu = { chol_res.transpose() * &crosscorr };
        let sigma = { postcorr - chol_res.transpose() * crosscorr };
        Ok((mu, sigma))
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::*;
    use faer::{mat, solvers::Solver, Faer, Mat};
    // use super::*;
    // use serde_json::Result;
    #[test]
    fn it_works() {
        let m1 = Mat::from_fn(3, 1, |i, j| i as f64 + j as f64);
        let a = mat![[4., 12., -16.], [12., 37., -43.], [-16., -43., 98f64],];
        let decomp = a.cholesky(faer::Side::Lower).unwrap();
        let sol = decomp.solve(&m1);
        let round = a * sol;
        for i in 0..3 {
            assert_approx_eq!(m1.get(i, 0), round.get(i, 0))
        }
        assert!(1 == 2);
    }
}
