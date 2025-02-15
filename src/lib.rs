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

#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Periodic,
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
T: Float 
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
where T: Float
{
    #[inline]
    fn len(&self) -> usize {
        self.length
    }
}

impl<T> GenericCosineIter<T>
where
    T: Float
{
    pub fn new(length: usize, window_type: WindowType, a0: T, a1: T, a2: T, a3: T, a4: T) -> Self {
        let len_float = match window_type {
            WindowType::Periodic => T::from(length).unwrap(),
            WindowType::Symmetric => T::from(length-1).unwrap(),
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
        self.a0 - self.a1 * (self.pi2 * x_float / self.len_float).cos() + self.a2 * (self.pi4 * x_float / self.len_float).cos()
            - self.a3 * (self.pi6 * x_float / self.len_float).cos() +  self.a4 * (self.pi8 * x_float / self.len_float).cos()
    }
}

/// Make the selected window function.
pub fn cosine_window<T>(length: usize, windowfunc: WindowFunction, window_type: WindowType) -> GenericCosineIter<T>
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
            T::from(0.355768).unwrap(),
            T::from(0.487396).unwrap(),
            T::from(0.144232).unwrap(),
            T::from(0.012604).unwrap(),
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
    use approx::assert_abs_diff_eq;

    // TODO extract reference windows to compare with, for exampel from numpy or matlab.

    #[test]
    fn test_blackman_harris() {
        let wnd_iter = cosine_window::<f64>(16, WindowFunction::BlackmanHarris, WindowType::Periodic);
        let wnd: Vec<f64> = wnd_iter.into_iter().collect();
        assert_abs_diff_eq!(wnd[0], 0.0, epsilon = 0.0001);
        assert_abs_diff_eq!(wnd[8], 1.0, epsilon = 0.000001);
        assert_abs_diff_eq!(wnd[15], 0.003, epsilon = 0.001);
    }

    #[test]
    fn test_hann() {
        let wnd_iter_per = cosine_window::<f64>(6, WindowFunction::Hann, WindowType::Periodic);
        let wnd_iter = cosine_window::<f64>(7, WindowFunction::Hann, WindowType::Symmetric);
        let hann_expected = vec![
            0.0,
            0.25,
            0.75,
            1.0,
            0.75,
            0.25,
            0.0,
        ];
        for (actual, expected) in wnd_iter_per.into_iter().zip(&hann_expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 0.000001);
        }
        for (actual, expected) in wnd_iter.into_iter().zip(&hann_expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 0.000001);
        }
    }
}
