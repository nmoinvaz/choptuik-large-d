//! A block of the exact Jacobian from the column tasks that reach it: J[rows, cols] in any precision, bit for bit the
//! entries of `model::jacobian_in` there. hp/reseed.py's rung block (the horizon block's m and a rows by the 2M
//! horizon-data columns) then costs the horizon block's columns only, not the centre series' and the segments'.
use crate::model::{Ctx, Dss};
use multidouble::precision::Precision;
use multidouble::simd::Native;
use multishoot::jacobian::Seed;
use multishoot::Equations;
use rayon::prelude::*;
use std::collections::HashMap;

/// J[rows[i], cols[j]] at z in P, as PARTS part arrays of rows.len() x cols.len() entries, row-major, bit for bit the
/// entries of `jacobian_in::<P>` (multishoot's `Shooting::jacobian`) there. Only the column tasks (`jac_tasks`) whose
/// block's rows meet `rows` and whose columns meet `cols` are evaluated, by `jac_column` with the globals and the
/// preparation `jacobian` uses, and each entry is assembled in the order of multishoot's `assemble`: the column's
/// value (zero where no task writes), the target's -1, then the extra (phase) row's weights. A context with frozen
/// rows (Ctx::freeze) or a row scaling is refused, since their projection and scaling mix a column's entries.
pub fn jacobian_sub_in<P: Precision<Native>, const B: usize>(
    ctx: &Ctx<B>,
    z: &[&[f64]],
    rows: &[usize],
    cols: &[usize],
    par: bool,
) -> Result<Vec<Vec<f64>>, String> {
    let sh = ctx.shooting_in::<P>();
    sh.check_z::<P>(z)?;
    if !sh.is_square() {
        return Err(format!("jacobian_sub: {} rows for {} unknowns", sh.rows(), sh.len()));
    }
    if !sh.projected.is_empty() || !sh.row_scale.is_empty() {
        return Err("jacobian_sub: frozen rows (kfreeze) or a row scaling mix a column's entries; use the full Jacobian".into());
    }
    let l = sh.len();
    if let Some(&bad) = rows.iter().chain(cols).find(|&&i| i >= l) {
        return Err(format!("jacobian_sub: index {bad} out of range for {l} unknowns"));
    }
    let m = Ctx::<B>::M;
    let nr_block = <Dss<'static, B> as Equations<B>>::NF * m;
    let k = sh.k();
    // the block whose rows hold row r; None for the extra rows
    let block_of = |r: usize| (0..=k).find(|&b| r >= sh.row_block(b) && r < sh.row_block(b) + nr_block);
    let tasks = sh.jac_tasks();
    // (block, column) -> (task, the column's place among the task's columns)
    let mut owner: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for (ti, &(b, seed)) in tasks.iter().enumerate() {
        for (jj, c) in sh.seed_columns(b, seed).into_iter().enumerate() {
            owner.insert((b, c), (ti, jj));
        }
    }
    let rb: Vec<Option<usize>> = rows.iter().map(|&r| block_of(r)).collect();
    let owner = &owner;
    let mut need: Vec<usize> = rb
        .iter()
        .flatten()
        .flat_map(|&b| cols.iter().filter_map(move |&c| owner.get(&(b, c)).map(|&(ti, _)| ti)))
        .collect();
    need.sort_unstable();
    need.dedup();
    let g = sh.globals::<P>(z);
    // multishoot's `prep` (pub(crate)) is this outside Control::InfalliblePrepare, which hp never sets
    let prep = sh.eq.try_prepare(z[0], z[1])?;
    let column = |ti: &usize| sh.jac_column::<P>(z, &g, &prep, tasks[*ti]);
    let vals: Vec<Vec<P::Scalar>> = if par { need.par_iter().map(column).collect() } else { need.iter().map(column).collect() };
    let at: HashMap<usize, usize> = need.iter().enumerate().map(|(i, &ti)| (ti, i)).collect();
    let (nr, nc) = (rows.len(), cols.len());
    let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; nr * nc]).collect();
    let re = sh.row_extra();
    let mut v = vec![0.0; P::PARTS];
    let set = |v: &mut [f64], x: P::Scalar| {
        for (q, vq) in v.iter_mut().enumerate() {
            *vq = P::part(x, q);
        }
    };
    for (i, &r) in rows.iter().enumerate() {
        for (j, &c) in cols.iter().enumerate() {
            v.iter_mut().for_each(|x| *x = 0.0);
            if let Some(b) = rb[i] {
                let r0 = sh.row_block(b);
                let rr = r - r0;
                if let Some(&(ti, jj)) = owner.get(&(b, c)) {
                    // a pointwise task's column jj holds point jj of every output field only (multishoot's assemble)
                    if !matches!(tasks[ti].1, Seed::Points(_)) || rr % m == jj {
                        set(&mut v, vals[at[&ti]][rr]);
                    }
                }
                if c == sh.block_target(b) + rr {
                    let s = P::from_raw(&v) + (-P::ONE);
                    set(&mut v, s);
                }
            } else if r >= re && r - re < sh.extra.len() {
                for &(ci, w) in &sh.extra[r - re].terms {
                    if ci == c {
                        let s = P::from_raw(&v) + P::from_dd(w);
                        set(&mut v, s);
                    }
                }
            }
            for (q, oq) in out.iter_mut().enumerate() {
                oq[i * nc + j] = v[q];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::jacobian_sub_in;
    use crate::model::{jacobian_in, Ctx};
    use crate::qd_residual::synth;
    use multidouble::precision::{Double, Octo, Precision, Quad};
    use multidouble::simd::Native;

    /// z in P::PARTS parts: the synthetic solution moved off it in its leading part, with nonzero lower parts
    fn state(zh: &[f64], zl: &[f64], parts: usize) -> Vec<Vec<f64>> {
        let l = zh.len();
        let mut z = vec![zh.iter().enumerate().map(|(i, x)| if i + 1 < l { x + 1e-3 * (0.7 * i as f64).sin() } else { *x }).collect::<Vec<_>>(), zl.to_vec()];
        for q in 2..parts {
            z.push(zh.iter().enumerate().map(|(i, x)| x * (0.3 + 0.01 * i as f64) * 2f64.powi(-53 * q as i32)).collect());
        }
        z
    }

    /// for each (rows, cols) case, the entries of J[rows, cols] whose bits differ from the full Jacobian's in any part;
    /// the full Jacobian formed once. ulp: the control, one ulp up in the last part of each block's first entry.
    fn differ<P: Precision<Native>>(ctx: &Ctx<1>, z: &[Vec<f64>], cases: &[(&[usize], &[usize])], ulp: bool) -> Vec<usize> {
        let zv: Vec<&[f64]> = z.iter().map(|p| &p[..]).collect();
        let (full, ld) = jacobian_in::<P, 1>(ctx, &zv, true).unwrap();
        cases
            .iter()
            .map(|&(rows, cols)| {
                let mut sub = jacobian_sub_in::<P, 1>(ctx, &zv, rows, cols, true).unwrap();
                if ulp {
                    let x = &mut sub[P::PARTS - 1][0];
                    *x = f64::from_bits(x.to_bits() + 1);
                }
                let nc = cols.len();
                let mut n = 0;
                for (i, &r) in rows.iter().enumerate() {
                    for (j, &c) in cols.iter().enumerate() {
                        n += (0..P::PARTS).any(|q| sub[q][i * nc + j].to_bits() != full[q][c * ld + r].to_bits()) as usize;
                    }
                }
                n
            })
            .collect()
    }

    /// Pass (fixed before the first run): on the synthetic context moved off its solution, J[rows, cols] from
    /// jacobian_sub_in is the full Jacobian's (jacobian_in) in every part of every entry, in double-, quad- and
    /// octo-double, for (a) the horizon block's m and a rows by its m_h and a_h columns, the re-seed's block, and (b)
    /// every row by columns that hold the targets' -1 (the last node's), the phase row's weights (p1's), Delta, a
    /// segment's input and a repeated index, in no order. Control, which must differ by one entry in each block: one
    /// ulp of the block's first entry, in its last part (quad-double).
    #[test]
    fn jacobian_sub_bitwise() {
        let (ctx, zh, zl) = synth();
        let (l, m, k) = (ctx.len(), Ctx::<1>::M, ctx.k());
        let r0 = 4 * m * k;
        let rung_rows: Vec<usize> = (r0 + m..r0 + 3 * m).collect();
        let rung_cols: Vec<usize> = (ctx.off_node(k)..ctx.off_node(k) + 2 * m).collect();
        let all_rows: Vec<usize> = (0..l).rev().chain([3, 3]).collect();
        let t = ctx.off_node(k - 1);
        let mix_cols: Vec<usize> = vec![l - 1, t + 2, 1, t + m + 5, ctx.off_node(1) + 3, 0, t + 2, ctx.off_node(k) + 1];
        let cases: [(&[usize], &[usize]); 2] = [(&rung_rows, &rung_cols), (&all_rows, &mix_cols)];
        let mut got = differ::<Double>(&ctx, &state(&zh, &zl, 2), &cases, false);
        got.extend(differ::<Quad>(&ctx, &state(&zh, &zl, 4), &cases, false));
        got.extend(differ::<Octo>(&ctx, &state(&zh, &zl, 8), &cases, false));
        let control = differ::<Quad>(&ctx, &state(&zh, &zl, 4), &cases, true);
        println!("J[rows, cols] against the full Jacobian, entries that differ (dd, qd, od; rung block, mixed): {got:?}; control {control:?}");
        assert_eq!(got, vec![0; 6], "a block entry is not the full Jacobian's");
        assert!(control.iter().all(|&n| n == 1), "the control did not change exactly one entry");
    }
}
