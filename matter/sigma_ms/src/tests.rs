//! Unit checks of the pieces that planted.py does not isolate: the order-0 sigma and gamma.
use crate::model::sin_cos_field;
use multidouble::dd::{split, DD};
use multidouble::field::Fl;
use multidouble::native::DV;
use multishoot::periodic::sin_cos;

fn field(v: &[DD]) -> Fl<DV, 1> {
    let (h, l) = split(v);
    Fl::load(&h, &l)
}

/// sigma = sin(kappa psi)/kappa and gamma = cos(kappa psi) against multishoot's double-double sin_cos for
/// |kappa psi| up to 1, to 1e-31; the control (kappa^2 off by 2^-40) must miss by more than 1e-15
#[test]
fn sin_cos_field_matches_sin_cos() {
    let k2 = DD::from_f64(0.25);
    let kap = DD::from_f64(0.5);
    let psi: Vec<DD> = (0..8).map(|j| DD::from_f64(-2.0 + 0.55 * j as f64)).collect();
    let (s, c) = sin_cos_field::<DV, 1>(&field(&psi), k2);
    let mut worst: f64 = 0.0;
    for j in 0..8 {
        let (sr, cr) = sin_cos(kap * psi[j]);
        worst = worst.max((s.get(j) - sr / kap).hi.abs()).max((c.get(j) - cr).hi.abs());
    }
    assert!(worst < 1e-31, "worst {worst:e}");
    let (s2, _) = sin_cos_field::<DV, 1>(&field(&psi), k2 + k2.mul_f(2f64.powi(-40)));
    let miss = (0..8).map(|j| (s2.get(j) - sin_cos(kap * psi[j]).0 / kap).hi.abs()).fold(0.0, f64::max);
    assert!(miss > 1e-15, "control missed by {miss:e} only");
}

/// at kappa = 0, sigma = psi and gamma = 1 bit for bit (the l = 1 multiplet)
#[test]
fn sin_cos_field_at_kappa_zero() {
    let psi: Vec<DD> = (0..8).map(|j| DD::new(0.1 + 0.37 * j as f64, 1e-18 * j as f64)).collect();
    let (s, c) = sin_cos_field::<DV, 1>(&field(&psi), DD::ZERO);
    for j in 0..8 {
        assert_eq!((s.get(j).hi, s.get(j).lo), (psi[j].hi, psi[j].lo));
        assert_eq!((c.get(j).hi, c.get(j).lo), (1.0, 0.0));
    }
}
