#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style
)]
use std::{cmp::Reverse, collections::BinaryHeap};

use faer::{solvers::Solver, Faer, IntoFaer, IntoNalgebra, Mat};
use nalgebra::{Dyn, Matrix, VecStorage};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

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
        let n = x1.len();
        // let autocorr: Mat<f64> = self
        //     .process
        //     .autocorr
        //     .view_range(.., ..)
        //     .into_faer()
        //     .to_owned();
        let autocorr = Mat::from_fn(n, n, |i, j| (self.metric)(&x1[i], &x1[j]));
        let crosscorr = Mat::from_fn(x1.len(), x2.len(), |i, j| (self.metric)(&x1[i], &x2[j]));
        let postcorr = Mat::from_fn(x2.len(), x2.len(), |i, j| (self.metric)(&x2[i], &x2[j]));
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

    pub fn interpolate_one(&self, x2: &T) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
        let n = &self.process.inputs.len();
        let mut heap = BinaryHeap::with_capacity(n + 1);

        self.process
            .inputs
            .iter()
            .map(|val| NotNan::new((self.metric)(val, x2)).expect("NaN from metric"))
            .enumerate()
            .for_each(|(i, num)| {
                heap.push(Reverse((num, i)));
            });

        let indices: Vec<usize> = heap.into_vec().iter().map(|&Reverse((_, i))| i).collect();
        let x1: Vec<&T> = indices.iter().map(|i| &self.process.inputs[*i]).collect();
        let y1: Vec<&f64> = indices.iter().map(|i| &self.process.res[*i]).collect();
        let n = x1.len();
        // let autocorr: Mat<f64> = self
        //     .process
        //     .autocorr
        //     .view_range(.., ..)
        //     .into_faer()
        //     .to_owned();
        let autocorr = Mat::from_fn(n, n, |i, j| (self.metric)(&x1[i], &x1[j]));
        let crosscorr = Mat::from_fn(n, 1, |i, _| (self.metric)(&x1[i], &x2));
        let postcorr = Mat::from_fn(1, 1, |_, _| (self.metric)(&x2, &x2));
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

    // restrict the matrix to invert to be the n (TBD) closest values to the desired value
    // limitation of only infering one value at a time
    pub fn smart_interpolate(&self, x2: &T) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
        let n = 6561;

        let mut heap = BinaryHeap::with_capacity(n + 1);

        self.process
            .inputs
            .iter()
            .map(|val| NotNan::new((self.metric)(val, x2)).expect("NaN from metric"))
            .enumerate()
            .for_each(|(i, num)| {
                heap.push(Reverse((num, i)));
                if heap.len() > n {
                    heap.pop();
                }
            });

        let indices: Vec<usize> = heap.into_vec().iter().map(|&Reverse((_, i))| i).collect();

        println!("sorted");
        let x1: Vec<&T> = indices.iter().map(|i| &self.process.inputs[*i]).collect();
        let y1: Vec<&f64> = indices.iter().map(|i| &self.process.res[*i]).collect();

        let autocorr = Mat::from_fn(n, n, |i, j| (self.metric)(&x1[i], &x1[j]));
        let crosscorr = Mat::from_fn(n, 1, |i, _| (self.metric)(&x1[i], &x2));
        let postcorr = Mat::from_fn(1, 1, |_, _| (self.metric)(&x2, &x2));
        let y1 = Mat::from_fn(n, 1, |i, _| *y1[i]);

        let chol_res = match autocorr.cholesky(faer::Side::Lower) {
            Ok(value) => value.solve(&crosscorr),
            Err(_) => return Err(ProcessError::CholeskyFaiure),
        };

        let mu = { chol_res.transpose() * &y1 };
        let sigma = { postcorr - chol_res.transpose() * crosscorr };
        Ok((mu, sigma))
    }

    pub fn dyn_smart_interpolate(
        &self,
        x2: &T,
        n: usize,
    ) -> Result<(Mat<f64>, Mat<f64>), ProcessError> {
        // let n = 6561;

        let mut heap = BinaryHeap::with_capacity(n + 1);

        self.process
            .inputs
            .iter()
            .map(|val| NotNan::new((self.metric)(val, x2)).expect("NaN from metric"))
            .enumerate()
            .for_each(|(i, num)| {
                heap.push(Reverse((num, i)));
                if heap.len() > n {
                    heap.pop();
                }
            });

        let indices: Vec<usize> = heap.into_vec().iter().map(|&Reverse((_, i))| i).collect();

        println!("sorted");
        let x1: Vec<&T> = indices.iter().map(|i| &self.process.inputs[*i]).collect();
        let y1: Vec<&f64> = indices.iter().map(|i| &self.process.res[*i]).collect();

        let autocorr = Mat::from_fn(n, n, |i, j| (self.metric)(&x1[i], &x1[j]));
        let crosscorr = Mat::from_fn(n, 1, |i, _| (self.metric)(&x1[i], &x2));
        let postcorr = Mat::from_fn(1, 1, |_, _| (self.metric)(&x2, &x2));
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
//         // let m1 = Mat::from_fn(3, 1, |i, j| i as f64 + j as f64);
//         // let a = mat![[4., 12., -16.], [12., 37., -43.], [-16., -43., 98f64],];
//         // let decomp = a.cholesky(faer::Side::Lower).unwrap();
//         // let sol = decomp.solve(&m1);
//         // let round = a * sol;
//         // for i in 0..3 {
//         //     assert_approx_eq!(m1.get(i, 0), round.get(i, 0))
//         // }
//         // assert!(1 == 2);
//         let mut list = vec![
//             NotNan::new(5.).unwrap(),
//             NotNan::new(6.).unwrap(),
//             NotNan::new(2.).unwrap(),
//             NotNan::new(511.).unwrap(),
//             NotNan::new(23.).unwrap(),
//             NotNan::new(1.).unwrap(),
//             NotNan::new(8.).unwrap(),
//         ];
//         // list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
//         // list.iter();
//         // println!("{:?}", list);
//         // panic!()
//         let n = 3; // Replace 3 with any number you want
//         let mut heap = BinaryHeap::new();

//         list.iter().enumerate().for_each(|(i, &num)| {
//             heap.push(Reverse((num, i)));
//             if heap.len() > n {
//                 heap.pop();
//             }
//         });

//         let mut largest = heap.into_vec();
//         largest.sort();

//         let indices_of_largest: Vec<usize> = largest.iter().map(|&Reverse((_, i))| i).collect();
//         println!("{:?}", indices_of_largest);
//         panic!()
//     }
// }
