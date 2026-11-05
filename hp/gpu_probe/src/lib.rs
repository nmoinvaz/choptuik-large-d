//! One lane operation per exported function, so that `count.py` can count its instructions in the
//! disassembly of the cdylib (branch-free lane code: the static count is the dynamic one).
use multidouble::native::{DV, QV};

#[no_mangle]
#[inline(never)]
pub extern "C" fn k_dv_add(a: &DV, b: &DV, o: &mut DV) {
    *o = *a + *b;
}
#[no_mangle]
#[inline(never)]
pub extern "C" fn k_dv_mul(a: &DV, b: &DV, o: &mut DV) {
    *o = *a * *b;
}
#[no_mangle]
#[inline(never)]
pub extern "C" fn k_dv_axpy(a: &DV, b: &DV, c: &DV, o: &mut DV) {
    *o = *a + *b * *c;
}
#[no_mangle]
#[inline(never)]
pub extern "C" fn k_qv_add(a: &QV, b: &QV, o: &mut QV) {
    *o = *a + *b;
}
#[no_mangle]
#[inline(never)]
pub extern "C" fn k_qv_mul(a: &QV, b: &QV, o: &mut QV) {
    *o = *a * *b;
}
#[no_mangle]
#[inline(never)]
pub extern "C" fn k_qv_axpy(a: &QV, b: &QV, c: &QV, o: &mut QV) {
    *o = *a + *b * *c;
}
