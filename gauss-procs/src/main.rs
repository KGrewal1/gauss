use dyn_stack::{GlobalPodBuffer, PodStack};
use faer::{mat, Faer, Mat}; //, solvers::Solver
use faer_cholesky::llt::{
    // compute::cholesky_in_place,
    reconstruct::{reconstruct_lower, reconstruct_lower_req},
    update::{
        delete_rows_and_cols_clobber, delete_rows_and_cols_clobber_req,
        insert_rows_and_cols_clobber, insert_rows_and_cols_clobber_req,
    },
};
use faer_core::{MatRef, Parallelism};
use gauss_lib::*;
use itertools::Itertools;
use std::time::Instant;

fn lim_nonpoly(x: &TwoDpoint) -> f64 {
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

fn reconstruct_matrix(cholesky_factor: MatRef<f64>) -> Mat<f64> {
    let n = cholesky_factor.nrows();

    let mut a_reconstructed = Mat::zeros(n, n);
    reconstruct_lower(
        a_reconstructed.as_mut(),
        cholesky_factor,
        Parallelism::Rayon(0),
        PodStack::new(&mut GlobalPodBuffer::new(
            reconstruct_lower_req::<f64>(n).unwrap(),
        )),
    );

    a_reconstructed
}

#[allow(unreachable_code)]
fn main() {
    // println!("{}", lim_nonpoly((0., 0.)));
    // let mut matrix: Mat<f64> = mat![[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.],];
    let matrix: Mat<f64> = mat![[4., 12.], [12., 37.]];
    println!("{:?}", matrix.get(0..2, 1));
    // let mut decomp = matrix.cholesky(faer::Side::Lower).unwrap();
    // cholesky_in_place(
    //     matrix.as_mut(),
    //     Parallelism::Rayon(0),
    //     PodStack::new(&mut []),
    //     Default::default(),
    // )
    // .unwrap();
    // testing to dynamically update cholesky matrix
    let matrix = matrix.cholesky(faer::Side::Lower).unwrap();
    // println!("{:?}", matrix);
    let mut matrix = matrix.compute_l();
    println!("matrix {:?}", matrix);
    println!("matrix {:?}", reconstruct_matrix(matrix.as_ref()));
    matrix.resize_with(3, 3, |_, _| 0.);
    println!("Here");
    let mut new_col = mat![[-16., -43., 98.],].transpose().to_owned();
    insert_rows_and_cols_clobber(
        matrix.as_mut(),
        2,
        new_col.as_mut(),
        Parallelism::Rayon(8),
        PodStack::new(&mut GlobalPodBuffer::new(
            insert_rows_and_cols_clobber_req::<f64>(4, Parallelism::Rayon(8)).unwrap(),
        )),
    )
    .unwrap();
    println!("matrix {:?}", matrix);
    println!("matrix {:?}", reconstruct_matrix(matrix.as_ref()));
    delete_rows_and_cols_clobber(
        matrix.as_mut(),
        &mut [0],
        Parallelism::Rayon(8),
        PodStack::new(&mut GlobalPodBuffer::new(
            delete_rows_and_cols_clobber_req::<f64>(3, 1, Parallelism::Rayon(8)).unwrap(),
        )),
    );

    println!("matrix {:?}", matrix);
    println!("matrix {:?}", reconstruct_matrix(matrix.as_ref()));
    panic!();
    let n: usize = 80;
    let range: Vec<f64> = (0..(n + 1)).map(|i| i as f64 / (n as f64)).collect();
    let inputs: Vec<TwoDpoint> = range
        .clone()
        .into_iter()
        .cartesian_product(range)
        .map(|(i, j)| TwoDpoint(i, j))
        .collect();
    let outputs: Vec<f64> = inputs.iter().map(lim_nonpoly).collect();
    println!(
        "metric {}",
        Kernel::metric(&inputs[15], &inputs[20], &[1750.])
    );
    println!(
        "deriv {:?}",
        Kernel::deriv(&inputs[15], &inputs[20], &[1750.])
    );

    // println!("{:?}, {:?}", &inputs[100], &inputs[99]);
    // println!("{}", Kernel::metric(&inputs[100], &inputs[99], &[1750.]));
    // println!("{}", inputs.len());
    // println!("grad original {:?}\n", gradient(1750., &inputs, &outputs));
    let proc = GaussProcs::new(inputs, outputs, 0., [1750.]).unwrap();

    // println!("{}", proc.log_marginal_likelihood().unwrap());

    println!("grad new {:?}", proc.gradient());

    // let now = Instant::now();
    // println!(
    //     "{:?}",
    //     proc.smart_interpolate(&TwoDpoint(0.215, 0.255)).unwrap()
    // );
    // let elapsed = now.elapsed();
    // println!("Elapsed, trunc: {:.2?}", elapsed);

    let now = Instant::now();
    println!(
        "{:?}",
        proc.dyn_smart_interpolate(&TwoDpoint(0.215, 0.255), 6561)
            .unwrap()
    );
    let elapsed = now.elapsed();
    println!("Elapsed, trunc: {:.2?}", elapsed);

    let now = Instant::now();
    println!("{:?}", proc.interpolate(&[TwoDpoint(0.215, 0.255)]));
    let elapsed = now.elapsed();
    println!("Elapsed, non trunc: {:.2?}", elapsed);

    // let now = Instant::now();

    // println!(
    //     "{:?}",
    //     proc.interpolate_one(&TwoDpoint(0.215, 0.255)).unwrap()
    // );
    // let elapsed = now.elapsed();
    // println!("Elapsed, trunc: {:.2?}", elapsed);

    // let predicted = *proc
    //     .interpolate_one(&TwoDpoint(0.215, 0.255))
    //     .unwrap()
    //     .0
    //     .get(0, 0);

    let predicted = *(proc
        .dyn_smart_interpolate(&TwoDpoint(0.215, 0.255), 100)
        .unwrap()
        .0)
        .get(0, 0);

    // println!("{:?}", lim_nonpoly(&TwoDpoint(0.215, 0.255)));
    println!("{}", lim_nonpoly(&TwoDpoint(0.215, 0.255)) - predicted)
}
