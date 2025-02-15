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
    Bartlett,
    Triangular,
    // Rectangular
    // there are many more, what makes sense to include?
}

enum WindowFamily {
    Cosine,
    Triangular,
}

/// Specify the symmetry of a window function.
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Generate a periodic window, often used in spectral analysis.
    Periodic,
    /// Generate a symmetric window, often used in filter design.
    Symmetric,
}

pub struct GenericWindowIter<T> {
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
    family: WindowFamily,
}

impl<T> Iterator for GenericWindowIter<T>
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

impl<T> ExactSizeIterator for GenericWindowIter<T>
where
    T: Float,
{
    #[inline]
    fn len(&self) -> usize {
        self.length
    }
}

impl<T> GenericWindowIter<T>
where
    T: Float,
{
    pub fn new_cosine(
        length: usize,
        window_type: WindowType,
        a0: T,
        a1: T,
        a2: T,
        a3: T,
        a4: T,
    ) -> Self {
        let len_float = match window_type {
            WindowType::Periodic => T::from(length).unwrap(),
            WindowType::Symmetric => T::from(length - 1).unwrap(),
        };
        GenericWindowIter {
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
            family: WindowFamily::Cosine,
        }
    }

    fn new_triangular(length: usize, window_type: WindowType, len_offset: usize) -> Self {
        let len_adjusted = match window_type {
            WindowType::Periodic => length,
            WindowType::Symmetric => length - 1,
        };
        let len_float = T::from(len_adjusted).unwrap();
        let a0 = len_float / T::from(2).unwrap();
        let a1 = if len_offset > 0 {
            (len_float + T::from(len_offset + 1 - len_adjusted % 2).unwrap()) / T::from(2).unwrap()
        } else {
            a0
        };
        GenericWindowIter {
            a0,
            a1,
            a2: T::zero(),
            a3: T::zero(),
            a4: T::zero(),
            index: 0,
            length,
            len_float,
            pi2: T::zero(),
            pi4: T::zero(),
            pi6: T::zero(),
            pi8: T::zero(),
            family: WindowFamily::Triangular,
        }
    }

    fn calc_at_index(&self) -> T {
        let x_float = T::from(self.index).unwrap();
        match self.family {
            WindowFamily::Cosine => {
                self.a0 - self.a1 * (self.pi2 * x_float / self.len_float).cos()
                    + self.a2 * (self.pi4 * x_float / self.len_float).cos()
                    - self.a3 * (self.pi6 * x_float / self.len_float).cos()
                    + self.a4 * (self.pi8 * x_float / self.len_float).cos()
            }
            WindowFamily::Triangular => T::one() - ((x_float - self.a0) / self.a1).abs(),
        }
    }
}

/// Make the selected window function.
pub fn window<T>(
    length: usize,
    windowfunc: WindowFunction,
    window_type: WindowType,
) -> GenericWindowIter<T>
where
    T: Float,
{
    match windowfunc {
        WindowFunction::BlackmanHarris => GenericWindowIter::new_cosine(
            length,
            window_type,
            T::from(0.35875).unwrap(),
            T::from(0.48829).unwrap(),
            T::from(0.14128).unwrap(),
            T::from(0.01168).unwrap(),
            T::zero(),
        ),
        WindowFunction::Blackman => GenericWindowIter::new_cosine(
            length,
            window_type,
            T::from(0.42).unwrap(),
            T::from(0.5).unwrap(),
            T::from(0.08).unwrap(),
            T::zero(),
            T::zero(),
        ),
        WindowFunction::Hamming => GenericWindowIter::new_cosine(
            length,
            window_type,
            T::from(0.53836).unwrap(),
            T::from(0.46164).unwrap(),
            T::zero(),
            T::zero(),
            T::zero(),
        ),
        WindowFunction::Hann => GenericWindowIter::new_cosine(
            length,
            window_type,
            T::from(0.5).unwrap(),
            T::from(0.5).unwrap(),
            T::zero(),
            T::zero(),
            T::zero(),
        ),
        WindowFunction::Nuttall => GenericWindowIter::new_cosine(
            length,
            window_type,
            T::from(0.3635819).unwrap(),
            T::from(0.4891775).unwrap(),
            T::from(0.1365995).unwrap(),
            T::from(0.0106411).unwrap(),
            T::zero(),
        ),
        WindowFunction::BlackmanNutall => GenericWindowIter::new_cosine(
            length,
            window_type,
            T::from(0.3635819).unwrap(),
            T::from(0.4891775).unwrap(),
            T::from(0.1365995).unwrap(),
            T::from(0.0106411).unwrap(),
            T::zero(),
        ),
        WindowFunction::FlatTop => GenericWindowIter::new_cosine(
            length,
            window_type,
            T::from(0.21557895).unwrap(),
            T::from(0.41663158).unwrap(),
            T::from(0.277263158).unwrap(),
            T::from(0.083578947).unwrap(),
            T::from(0.006947368).unwrap(),
        ),
        WindowFunction::Bartlett => GenericWindowIter::new_triangular(length, window_type, 0),
        WindowFunction::Triangular => GenericWindowIter::new_triangular(length, window_type, 1),
    }
}

#[cfg(test)]
mod tests {
    extern crate approx;
    use crate::window;
    use crate::WindowFunction;
    use crate::WindowType;
    use num_traits::Float;
    use std::fmt::Debug;

    #[test]
    fn test_hann_odd() {
        let expected = vec![
            0.0, 0.0669873, 0.25, 0.5, 0.75, 0.9330127, 1.0, 0.9330127, 0.75, 0.5, 0.25, 0.0669873,
            0.0,
        ];
        check_window(WindowFunction::Hann, &expected);
    }

    #[test]
    fn test_hann_even() {
        let expected_even = vec![
            0.0,
            0.0572719871733951,
            0.21596762663442215,
            0.4397316598723384,
            0.6773024435212678,
            0.8742553740855505,
            0.985470908713026,
            0.985470908713026,
            0.8742553740855505,
            0.6773024435212679,
            0.43973165987233875,
            0.21596762663442215,
            0.05727198717339521,
            0.0,
        ];
        check_window(WindowFunction::Hann, &expected_even);
    }

    #[test]
    fn test_hamming_odd() {
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
        check_window(WindowFunction::Hamming, &expected);
    }

    #[test]
    fn test_hamming_even() {
        let expected = vec![
            0.0767199999999999,
            0.12959808031745212,
            0.2761185903190292,
            0.4827154469269326,
            0.7020598000543161,
            0.8839025017857071,
            0.9865855805965627,
            0.9865855805965627,
            0.8839025017857072,
            0.7020598000543162,
            0.4827154469269329,
            0.2761185903190292,
            0.12959808031745224,
            0.0767199999999999,
        ];
        check_window(WindowFunction::Hamming, &expected);
    }

    #[test]
    fn test_blackman_odd() {
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
        check_window(WindowFunction::Blackman, &expected);
    }

    #[test]
    fn test_blackman_even() {
        let expected = vec![
            -1.3877787807814457e-17,
            0.022717166911887535,
            0.10759923567101926,
            0.2820563144782543,
            0.5374215836675796,
            0.8038983085059763,
            0.9763073907652827,
            0.9763073907652828,
            0.8038983085059764,
            0.5374215836675799,
            0.2820563144782545,
            0.10759923567101926,
            0.022717166911887583,
            -1.3877787807814457e-17,
        ];
        check_window(WindowFunction::Blackman, &expected);
    }

    #[test]
    fn test_blackman_harris_odd() {
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
        check_window(WindowFunction::BlackmanHarris, &expected);
    }

    #[test]
    fn test_blackman_harris_even() {
        let expected = vec![
            6.0000000000001025e-05,
            0.005238996226589691,
            0.04261168680481082,
            0.1668602695128325,
            0.4158082954127571,
            0.7346347391691189,
            0.9666910128738909,
            0.966691012873891,
            0.7346347391691193,
            0.4158082954127572,
            0.16686026951283278,
            0.04261168680481082,
            0.005238996226589701,
            6.0000000000001025e-05,
        ];
        check_window(WindowFunction::BlackmanHarris, &expected);
    }

    #[test]
    fn test_nuttall_odd() {
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
        check_window(WindowFunction::Nuttall, &expected);
    }

    #[test]
    fn test_nuttall_even() {
        let expected = vec![
            0.0003628000000000381,
            0.006751452513864563,
            0.04759044606176561,
            0.17576128736842006,
            0.4253782120718732,
            0.7401569329915648,
            0.9674626189925116,
            0.9674626189925118,
            0.7401569329915649,
            0.4253782120718734,
            0.17576128736842025,
            0.04759044606176561,
            0.006751452513864572,
            0.0003628000000000381,
        ];
        check_window(WindowFunction::Nuttall, &expected);
    }

    #[test]
    fn test_flat_top_odd() {
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
        check_window(WindowFunction::FlatTop, &expected);
    }

    #[test]
    fn test_flat_top_even() {
        let expected = vec![
            -0.0004210510000000013,
            -0.00836446681676538,
            -0.04346351871731231,
            -0.06805774015843734,
            0.08261602096572271,
            0.5066288028080755,
            0.9321146024187164,
            0.9321146024187167,
            0.5066288028080761,
            0.08261602096572282,
            -0.06805774015843732,
            -0.04346351871731231,
            -0.008364466816765486,
            -0.0004210510000000013,
        ];
        check_window(WindowFunction::FlatTop, &expected);
    }

    #[test]
    fn test_bartlett_odd() {
        let expected = vec![
            0.0,
            0.16666666666666666,
            0.3333333333333333,
            0.5,
            0.6666666666666666,
            0.8333333333333334,
            1.0,
            0.8333333333333333,
            0.6666666666666667,
            0.5,
            0.33333333333333326,
            0.16666666666666674,
            0.0,
        ];
        check_window(WindowFunction::Bartlett, &expected);
    }

    #[test]
    fn test_bartlett_even() {
        let expected = vec![
            0.0,
            0.15384615384615385,
            0.3076923076923077,
            0.46153846153846156,
            0.6153846153846154,
            0.7692307692307693,
            0.9230769230769231,
            0.9230769230769231,
            0.7692307692307692,
            0.6153846153846154,
            0.46153846153846145,
            0.3076923076923077,
            0.15384615384615374,
            0.0,
        ];
        check_window(WindowFunction::Bartlett, &expected);
    }

    #[test]
    fn test_triangular_odd() {
        let expected = vec![
            0.14285714285714285,
            0.2857142857142857,
            0.42857142857142855,
            0.5714285714285714,
            0.7142857142857143,
            0.8571428571428571,
            1.0,
            0.8571428571428572,
            0.7142857142857142,
            0.5714285714285714,
            0.4285714285714286,
            0.2857142857142858,
            0.1428571428571428,
        ];
        check_window(WindowFunction::Triangular, &expected);
    }

    #[test]
    fn test_triangular_even() {
        let expected = vec![
            0.07142857142857142,
            0.21428571428571427,
            0.35714285714285715,
            0.5,
            0.6428571428571429,
            0.7857142857142857,
            0.9285714285714286,
            0.9285714285714286,
            0.7857142857142857,
            0.6428571428571429,
            0.5,
            0.35714285714285715,
            0.21428571428571427,
            0.07142857142857142,
        ];
        check_window(WindowFunction::Triangular, &expected);
    }

    fn check_window<T: Float + Debug>(wfunc: WindowFunction, sym_expected: &[T]) {
        let sym_len = sym_expected.len();
        let per_len = sym_len - 1;
        let iter_per = window::<T>(per_len, wfunc, WindowType::Periodic);
        let iter_sym = window::<T>(sym_len, wfunc, WindowType::Symmetric);
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
