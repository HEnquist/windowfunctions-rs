use num_traits::Float;

/// Different window functions that can be used to window the sinc function.
#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    /// Blackman
    Blackman,
    /// Blackman-Harris
    BlackmanHarris,
    /// Hamming
    Hamming,
    /// Hann
    Hann,
    /// Nutall
    Nuttall,
    /// Blackman-Nutall
    BlackmanNutall,
    /// Flat top of Matlab, https://www.mathworks.com/help/signal/ref/flattopwin.html
    FlatTop,
}

/// Specify the symmetry of a window function.
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Generate a periodic window, often used in spectral analysis.
    Periodic,
    /// Generate a symmetric window, often used in filter design.
    Symmetric,
}

pub struct GenericCosineIter<T> {
    length: usize,
    index: usize,
    len_float: T,
    a0: T,
    a1: T,
    a2: T,
    a3: T,
    a4: T,
    pi2: T,
    pi4: T,
    pi6: T,
    pi8: T,
}

impl<T> Iterator for GenericCosineIter<T>
where
    T: Float,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.index == self.length {
            return None;
        }
        let val = self.calc_at_index();
        self.index += 1;
        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for GenericCosineIter<T>
where
    T: Float,
{
    #[inline]
    fn len(&self) -> usize {
        self.length
    }
}

impl<T> GenericCosineIter<T>
where
    T: Float,
{
    pub fn new(length: usize, window_type: WindowType, a0: T, a1: T, a2: T, a3: T, a4: T) -> Self {
        let len_float = match window_type {
            WindowType::Periodic => T::from(length).unwrap(),
            WindowType::Symmetric => T::from(length - 1).unwrap(),
        };
        GenericCosineIter {
            a0,
            a1,
            a2,
            a3,
            a4,
            index: 0,
            length,
            len_float,
            pi2: T::from(core::f64::consts::PI).unwrap() * T::from(2.0).unwrap(),
            pi4: T::from(core::f64::consts::PI).unwrap() * T::from(4.0).unwrap(),
            pi6: T::from(core::f64::consts::PI).unwrap() * T::from(6.0).unwrap(),
            pi8: T::from(core::f64::consts::PI).unwrap() * T::from(8.0).unwrap(),
        }
    }
    fn calc_at_index(&self) -> T {
        let x_float = T::from(self.index).unwrap();
        self.a0 - self.a1 * (self.pi2 * x_float / self.len_float).cos()
            + self.a2 * (self.pi4 * x_float / self.len_float).cos()
            - self.a3 * (self.pi6 * x_float / self.len_float).cos()
            + self.a4 * (self.pi8 * x_float / self.len_float).cos()
    }
}

/// Make the selected window function.
pub fn cosine_window<T>(
    length: usize,
    windowfunc: WindowFunction,
    window_type: WindowType,
) -> GenericCosineIter<T>
where
    T: Float,
{
    match windowfunc {
        WindowFunction::BlackmanHarris => GenericCosineIter::new(
            length,
            window_type,
            T::from(0.35875).unwrap(),
            T::from(0.48829).unwrap(),
            T::from(0.14128).unwrap(),
            T::from(0.01168).unwrap(),
            T::zero(),
        ),
        WindowFunction::Blackman => GenericCosineIter::new(
            length,
            window_type,
            T::from(0.42).unwrap(),
            T::from(0.5).unwrap(),
            T::from(0.08).unwrap(),
            T::zero(),
            T::zero(),
        ),
        WindowFunction::Hamming => GenericCosineIter::new(
            length,
            window_type,
            T::from(0.53836).unwrap(),
            T::from(0.46164).unwrap(),
            T::zero(),
            T::zero(),
            T::zero(),
        ),
        WindowFunction::Hann => GenericCosineIter::new(
            length,
            window_type,
            T::from(0.5).unwrap(),
            T::from(0.5).unwrap(),
            T::zero(),
            T::zero(),
            T::zero(),
        ),
        WindowFunction::Nuttall => GenericCosineIter::new(
            length,
            window_type,
            T::from(0.3635819).unwrap(),
            T::from(0.4891775).unwrap(),
            T::from(0.1365995).unwrap(),
            T::from(0.0106411).unwrap(),
            T::zero(),
        ),
        WindowFunction::BlackmanNutall => GenericCosineIter::new(
            length,
            window_type,
            T::from(0.3635819).unwrap(),
            T::from(0.4891775).unwrap(),
            T::from(0.1365995).unwrap(),
            T::from(0.0106411).unwrap(),
            T::zero(),
        ),
        WindowFunction::FlatTop => GenericCosineIter::new(
            length,
            window_type,
            T::from(0.21557895).unwrap(),
            T::from(0.41663158).unwrap(),
            T::from(0.277263158).unwrap(),
            T::from(0.083578947).unwrap(),
            T::from(0.006947368).unwrap(),
        ),
    }
}

#[cfg(test)]
mod tests {
    extern crate approx;
    use crate::cosine_window;
    use crate::WindowFunction;
    use crate::WindowType;
    use num_traits::Float;
    use std::fmt::Debug;

    #[test]
    fn test_hann() {
        let hann_expected = vec![
            0.0, 0.0669873, 0.25, 0.5, 0.75, 0.9330127, 1.0, 0.9330127, 0.75, 0.5, 0.25, 0.0669873,
            0.0,
        ];
        check_cosine_window(WindowFunction::Hann, &hann_expected);
    }

    #[test]
    fn test_hamming() {
        let expected = vec![
            0.0767199999999999,
            0.1385680325969516,
            0.3075399999999998,
            0.53836,
            0.76918,
            0.9381519674030482,
            1.0,
            0.9381519674030483,
            0.7691800000000002,
            0.53836,
            0.3075400000000002,
            0.13856803259695172,
            0.0767199999999999,
        ];
        check_cosine_window(WindowFunction::Hamming, &expected);
    }

    #[test]
    fn test_blackman() {
        let expected = vec![
            -1.3877787807814457e-17,
            0.02698729810778064,
            0.1299999999999999,
            0.34,
            0.6299999999999999,
            0.8930127018922192,
            0.9999999999999999,
            0.8930127018922194,
            0.6300000000000002,
            0.34,
            0.1300000000000002,
            0.026987298107780687,
            -1.3877787807814457e-17,
        ];
        check_cosine_window(WindowFunction::Blackman, &expected);
    }

    #[test]
    fn test_blackman_harris() {
        let expected = vec![
            6.0000000000001025e-05,
            0.006518455586096459,
            0.05564499999999996,
            0.21747000000000008,
            0.5205749999999999,
            0.8522615444139033,
            1.0,
            0.8522615444139037,
            0.5205750000000002,
            0.21747000000000008,
            0.05564500000000015,
            0.006518455586096469,
            6.0000000000001025e-05,
        ];
        check_cosine_window(WindowFunction::BlackmanHarris, &expected);
    }

    #[test]
    fn test_nuttall() {
        let expected = vec![
            0.0003628000000000381,
            0.008241508040237797,
            0.06133449999999996,
            0.22698240000000006,
            0.5292298,
            0.8555217919597622,
            1.0,
            0.8555217919597622,
            0.5292298000000003,
            0.22698240000000006,
            0.06133450000000015,
            0.008241508040237806,
            0.0003628000000000381,
        ];
        check_cosine_window(WindowFunction::Nuttall, &expected);
    }

    #[test]
    fn test_flat_top() {
        let expected = vec![
            -0.0004210510000000013,
            -0.01007668729884861,
            -0.05126315599999999,
            -0.05473684,
            0.19821052999999986,
            0.7115503772988484,
            1.000000003,
            0.7115503772988487,
            0.1982105300000003,
            -0.05473684,
            -0.05126315600000008,
            -0.010076687298848712,
            -0.0004210510000000013,
        ];
        check_cosine_window(WindowFunction::FlatTop, &expected);
    }

    fn check_cosine_window<T: Float + Debug + approx::AbsDiffEq>(
        wfunc: WindowFunction,
        sym_expected: &[T],
    ) {
        let sym_len = sym_expected.len();
        let per_len = sym_len - 1;
        let iter_per = cosine_window::<T>(per_len, wfunc, WindowType::Periodic);
        let iter_sym = cosine_window::<T>(sym_len, wfunc, WindowType::Symmetric);
        for (idx, (actual, expected)) in iter_per.into_iter().zip(sym_expected).enumerate() {
            assert!(
                (actual - *expected).abs() < T::from(0.000001).unwrap(),
                "Diff at index {}, {:?} != {:?}",
                idx,
                actual,
                expected
            );
        }
        for (idx, (actual, expected)) in iter_sym.into_iter().zip(sym_expected).enumerate() {
            assert!(
                (actual - *expected).abs() < T::from(0.000001).unwrap(),
                "Diff at index {}, {:?} != {:?}",
                idx,
                actual,
                expected
            );
        }
    }
}
