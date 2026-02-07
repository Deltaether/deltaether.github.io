use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use rustfft::{FftPlanner, num_complex::Complex};

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[derive(Serialize, Deserialize)]
pub struct MethodResult {
    pub data: Vec<u8>,
    pub ai_probability: f32,  // 0.0 = definitely human, 1.0 = definitely AI
    pub metrics: Vec<(String, f32)>,
}

#[derive(Serialize, Deserialize)]
pub struct AnalysisResult {
    pub score: u32,
    pub max_score: u32,
    pub verdict: String,
    pub confidence: String,
    pub verdict_class: String,
    pub noise: MethodResult,
    pub high_pass: MethodResult,
    pub ela: MethodResult,
    pub posterize: MethodResult,
    pub channels: MethodResult,
    pub fft: MethodResult,
    pub fft_size: u32,
    pub noise_patterns: MethodResult,
    pub gradients: MethodResult,
    pub clip_space: MethodResult,
}

#[wasm_bindgen]
pub struct Analyzer {
    data: Vec<u8>,
    width: u32,
    height: u32,
    gray: Vec<f32>,
}

#[wasm_bindgen]
impl Analyzer {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Analyzer {
        let gray = Self::to_grayscale_static(&data, width, height);
        Analyzer { data, width, height, gray }
    }

    fn to_grayscale_static(data: &[u8], width: u32, height: u32) -> Vec<f32> {
        let len = (width * height) as usize;
        let mut gray = Vec::with_capacity(len);
        for i in 0..len {
            let idx = i * 4;
            let r = data[idx] as f32;
            let g = data[idx + 1] as f32;
            let b = data[idx + 2] as f32;
            gray.push(0.299 * r + 0.587 * g + 0.114 * b);
        }
        gray
    }

    fn get_gray(&self, x: i32, y: i32) -> f32 {
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return 0.0;
        }
        self.gray[(y as u32 * self.width + x as u32) as usize]
    }

    fn median_filter(&self, radius: i32) -> Vec<f32> {
        let len = (self.width * self.height) as usize;
        let mut result = vec![0.0; len];
        let size = ((radius * 2 + 1) * (radius * 2 + 1)) as usize;
        let mut values = Vec::with_capacity(size);

        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                values.clear();
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        values.push(self.get_gray(x + dx, y + dy));
                    }
                }
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                result[(y as u32 * self.width + x as u32) as usize] = values[values.len() / 2];
            }
        }
        result
    }

    fn gaussian_blur(&self, sigma: f32) -> Vec<f32> {
        let radius = (sigma * 2.0).ceil() as i32;
        let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
        let mut sum = 0.0;

        for i in -radius..=radius {
            let val = (-(i * i) as f32 / (2.0 * sigma * sigma)).exp();
            kernel.push(val);
            sum += val;
        }
        for v in kernel.iter_mut() {
            *v /= sum;
        }

        let len = (self.width * self.height) as usize;
        let mut temp = vec![0.0; len];
        let mut result = vec![0.0; len];

        // Horizontal pass
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                let mut val = 0.0;
                for (k, &weight) in kernel.iter().enumerate() {
                    let kx = k as i32 - radius;
                    val += self.get_gray(x + kx, y) * weight;
                }
                temp[(y as u32 * self.width + x as u32) as usize] = val;
            }
        }

        // Vertical pass
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                let mut val = 0.0;
                for (k, &weight) in kernel.iter().enumerate() {
                    let ky = k as i32 - radius;
                    let ny = (y + ky).clamp(0, self.height as i32 - 1);
                    val += temp[(ny as u32 * self.width + x as u32) as usize] * weight;
                }
                result[(y as u32 * self.width + x as u32) as usize] = val;
            }
        }
        result
    }

    fn calc_stats(data: &[f32]) -> (f32, f32) {
        let n = data.len() as f32;
        let mean: f32 = data.iter().sum::<f32>() / n;
        let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        (mean, variance.sqrt())
    }

    // Convert a value to probability using sigmoid-like curve
    // center: value where probability = 0.5
    // scale: how quickly it transitions (higher = sharper)
    // invert: if true, lower values = higher AI probability
    fn to_probability(value: f32, center: f32, scale: f32, invert: bool) -> f32 {
        let x = if invert {
            (center - value) * scale
        } else {
            (value - center) * scale
        };
        1.0 / (1.0 + (-x).exp())
    }

    fn amplify_to_u8(data: &[f32], factor: f32) -> Vec<u8> {
        data.iter()
            .map(|&v| (v * factor).min(255.0).max(0.0) as u8)
            .collect()
    }

    // Method 1: Noise Analysis
    // Anime/manga: AI has very uniform noise (no brush texture)
    // Human digital art has varied noise from brush strokes and layers
    fn analyze_noise(&self) -> MethodResult {
        let median = self.median_filter(2);
        let noise: Vec<f32> = self.gray.iter()
            .zip(median.iter())
            .map(|(&g, &m)| (g - m).abs())
            .collect();

        let (mean, std) = Self::calc_stats(&noise);
        // std < 12 = AI, std > 18 = human, sigmoid centered at 15
        let std_prob = Self::to_probability(std, 15.0, 0.3, true);
        // mean < 4 = AI, mean > 8 = human
        let mean_prob = Self::to_probability(mean, 6.0, 0.4, true);
        let ai_probability = (std_prob * 0.7 + mean_prob * 0.3).clamp(0.0, 1.0);
        
        let data = Self::amplify_to_u8(&noise, 10.0);

        MethodResult {
            data,
            ai_probability,
            metrics: vec![
                ("mean".into(), mean),
                ("std".into(), std),
            ],
        }
    }

    // Method 2: High Pass Filter
    // Anime/manga: AI tends to be too smooth or has artificial sharpening
    fn analyze_high_pass(&self) -> MethodResult {
        let blur = self.gaussian_blur(10.0);
        let high_pass: Vec<f32> = self.gray.iter()
            .zip(blur.iter())
            .map(|(&g, &b)| (g - b).abs())
            .collect();

        let (mean, std) = Self::calc_stats(&high_pass);
        // std < 18 = AI, std > 25 = human
        let ai_probability = Self::to_probability(std, 21.0, 0.2, true);
        let data = Self::amplify_to_u8(&high_pass, 5.0);

        MethodResult {
            data,
            ai_probability,
            metrics: vec![
                ("mean".into(), mean),
                ("std".into(), std),
            ],
        }
    }

    // Method 3: ELA (Error Level Analysis)
    // Anime/manga: AI has extremely uniform compression levels
    // Human art built in layers has varied ELA
    fn analyze_ela(&self) -> MethodResult {
        let len = (self.width * self.height) as usize;
        let mut ela = vec![0.0; len];

        for i in 0..len {
            let idx = i * 4;
            let r = self.data[idx] as f32;
            let g = self.data[idx + 1] as f32;
            let b = self.data[idx + 2] as f32;

            // Quantize (simulate JPEG Q95)
            let qr = (r / 8.0).round() * 8.0;
            let qg = (g / 8.0).round() * 8.0;
            let qb = (b / 8.0).round() * 8.0;

            ela[i] = ((r - qr).abs() + (g - qg).abs() + (b - qb).abs()) / 3.0;
        }

        let (mean, std) = Self::calc_stats(&ela);
        // mean < 5 = AI, mean > 8 = human
        let mean_prob = Self::to_probability(mean, 6.5, 0.4, true);
        // std < 6 = AI, std > 10 = human  
        let std_prob = Self::to_probability(std, 8.0, 0.3, true);
        let ai_probability = (mean_prob * 0.5 + std_prob * 0.5).clamp(0.0, 1.0);
        
        let data = Self::amplify_to_u8(&ela, 10.0);

        MethodResult {
            data,
            ai_probability,
            metrics: vec![
                ("mean".into(), mean),
                ("std".into(), std),
            ],
        }
    }

    // Method 4: Posterize / Color Banding
    // Anime/manga: Diffusion models create mathematically smooth gradients
    // Human artists use varied color transitions even with gradient tools
    fn analyze_posterize(&self) -> MethodResult {
        let len = (self.width * self.height) as usize;
        let mut diff = vec![0.0; len];

        for i in 0..len {
            let idx = i * 4;
            let r = self.data[idx] as f32;
            let g = self.data[idx + 1] as f32;
            let b = self.data[idx + 2] as f32;

            let pr = (r / 32.0).floor() * 32.0;
            let pg = (g / 32.0).floor() * 32.0;
            let pb = (b / 32.0).floor() * 32.0;

            diff[i] = ((r - pr).abs() + (g - pg).abs() + (b - pb).abs()) / 3.0;
        }

        let (mean, std) = Self::calc_stats(&diff);
        // mean < 8 = AI, mean > 12 = human
        let mean_prob = Self::to_probability(mean, 10.0, 0.3, true);
        // std < 6 = AI, std > 10 = human
        let std_prob = Self::to_probability(std, 8.0, 0.3, true);
        let ai_probability = (mean_prob * 0.6 + std_prob * 0.4).clamp(0.0, 1.0);
        
        let data = Self::amplify_to_u8(&diff, 5.0);

        MethodResult {
            data,
            ai_probability,
            metrics: vec![
                ("mean".into(), mean),
                ("std".into(), std),
            ],
        }
    }

    // Method 5: Channel Analysis
    // Anime/manga: AI generates RGB channels in sync (same latent space)
    // Human art has independent channel variation from color mixing
    fn analyze_channels(&self) -> MethodResult {
        let len = (self.width * self.height) as usize;
        let mut channel_diff = vec![0.0; len];

        for i in 0..len {
            let idx = i * 4;
            let r = self.data[idx] as f32;
            let g = self.data[idx + 1] as f32;
            let b = self.data[idx + 2] as f32;

            let rg = (r - g).abs();
            let rb = (r - b).abs();
            let gb = (g - b).abs();

            channel_diff[i] = (rg + rb + gb) / 3.0;
        }

        let (mean, std) = Self::calc_stats(&channel_diff);
        // std < 12 = AI, std > 18 = human
        let ai_probability = Self::to_probability(std, 15.0, 0.2, true);
        let data = Self::amplify_to_u8(&channel_diff, 3.0);

        MethodResult {
            data,
            ai_probability,
            metrics: vec![
                ("mean".into(), mean),
                ("std".into(), std),
            ],
        }
    }

    // Method 6: FFT Analysis
    // THE MOST RELIABLE for upscaled AI: cross pattern = Real-ESRGAN/Waifu2x
    fn analyze_fft(&self) -> (MethodResult, u32) {
        // Adaptive FFT size based on image dimensions
        // Use next power of 2 of the smaller dimension, clamped to reasonable range
        let min_dim = self.width.min(self.height);
        let size = Self::optimal_fft_size(min_dim);

        // Resize to square
        let resized = self.resize_gray(size, size);

        // Compute real FFT (O(n² log n) instead of O(n⁴))
        let fft_mag = self.compute_fft_real(&resized, size);

        // Detect cross pattern
        let cross_score = self.detect_cross_pattern(&fft_mag, size);

        // Normalize for display
        let max_val = fft_mag.iter().cloned().fold(0.0f32, f32::max);
        let data: Vec<u8> = fft_mag.iter()
            .map(|&v| ((v / max_val) * 255.0 * 2.0).min(255.0) as u8)
            .collect();

        // cross_score > 0.25 = likely AI, < 0.15 = likely human
        let ai_probability = Self::to_probability(cross_score, 0.20, 15.0, false);

        (MethodResult {
            data,
            ai_probability,
            metrics: vec![
                ("cross_score".into(), cross_score),
                ("fft_size".into(), size as f32),
            ],
        }, size)
    }

    // Calculate optimal FFT size (power of 2) based on image size
    // Larger images get higher resolution FFT for better pattern detection
    fn optimal_fft_size(min_dimension: u32) -> u32 {
        // Find next power of 2
        let next_pow2 = (min_dimension as f32).log2().ceil() as u32;
        let ideal_size = 1u32 << next_pow2;
        
        // Clamp between 256 (minimum for good detection) and 2048 (max for performance)
        // For very large images (4K+), cap at 2048 to avoid memory issues
        ideal_size.clamp(256, 2048)
    }

    fn resize_gray(&self, new_width: u32, new_height: u32) -> Vec<f32> {
        let x_ratio = self.width as f32 / new_width as f32;
        let y_ratio = self.height as f32 / new_height as f32;
        let mut result = vec![0.0; (new_width * new_height) as usize];

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 * x_ratio) as u32;
                let src_y = (y as f32 * y_ratio) as u32;
                result[(y * new_width + x) as usize] = 
                    self.gray[(src_y * self.width + src_x) as usize];
            }
        }
        result
    }

    // Real FFT using Cooley-Tukey algorithm - O(n² log n) instead of O(n⁴)
    fn compute_fft_real(&self, data: &[f32], size: u32) -> Vec<f32> {
        let n = size as usize;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);

        // Convert to complex and apply window function
        let mut buffer: Vec<Complex<f32>> = data.iter()
            .map(|&v| Complex::new(v / 255.0, 0.0))
            .collect();

        // 2D FFT: transform rows, then columns
        // Transform rows
        for y in 0..n {
            let start = y * n;
            let mut row: Vec<Complex<f32>> = buffer[start..start + n].to_vec();
            fft.process(&mut row);
            buffer[start..start + n].copy_from_slice(&row);
        }

        // Transform columns
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for x in 0..n {
            for y in 0..n {
                col[y] = buffer[y * n + x];
            }
            fft.process(&mut col);
            for y in 0..n {
                buffer[y * n + x] = col[y];
            }
        }

        // Compute magnitude and shift zero frequency to center
        let half = n / 2;
        let mut result = vec![0.0f32; n * n];
        
        for y in 0..n {
            for x in 0..n {
                let src_x = (x + half) % n;
                let src_y = (y + half) % n;
                let c = buffer[src_y * n + src_x];
                // Log scale for better visualization
                result[y * n + x] = (c.norm() + 1.0).ln();
            }
        }

        result
    }

    fn detect_cross_pattern(&self, fft_data: &[f32], size: u32) -> f32 {
        let center = size as i32 / 2;
        let width = 5;

        let mut cross_sum = 0.0;
        let mut cross_count = 0;
        let mut total_sum = 0.0;
        let mut total_count = 0;

        for y in 0..size as i32 {
            for x in 0..size as i32 {
                let val = fft_data[(y as u32 * size + x as u32) as usize];
                total_sum += val;
                total_count += 1;

                let on_h = (y - center).abs() < width && x != center;
                let on_v = (x - center).abs() < width && y != center;

                if on_h || on_v {
                    cross_sum += val;
                    cross_count += 1;
                }
            }
        }

        let cross_mean = cross_sum / cross_count as f32;
        let total_mean = total_sum / total_count as f32;

        cross_mean / (total_mean + 0.001)
    }

    // Method 7: Noise Patterns (AI vs Brush)
    fn analyze_noise_patterns(&self) -> MethodResult {
        let len = (self.width * self.height) as usize;
        let mut noise = vec![0.0; len];

        // Laplacian for texture extraction
        for y in 1..(self.height as i32 - 1) {
            for x in 1..(self.width as i32 - 1) {
                let idx = (y as u32 * self.width + x as u32) as usize;
                let laplacian = 
                    -self.get_gray(x, y - 1) +
                    -self.get_gray(x - 1, y) +
                    4.0 * self.get_gray(x, y) +
                    -self.get_gray(x + 1, y) +
                    -self.get_gray(x, y + 1);
                noise[idx] = laplacian.abs();
            }
        }

        // Calculate directional bias
        let mut h_energy = 0.0;
        let mut v_energy = 0.0;
        let mut d1_energy = 0.0;
        let mut d2_energy = 0.0;

        for y in 1..(self.height as i32 - 1) {
            for x in 1..(self.width as i32 - 1) {
                let gx = self.get_gray(x + 1, y) - self.get_gray(x - 1, y);
                let gy = self.get_gray(x, y + 1) - self.get_gray(x, y - 1);
                let gd1 = self.get_gray(x + 1, y - 1) - self.get_gray(x - 1, y + 1);
                let gd2 = self.get_gray(x - 1, y - 1) - self.get_gray(x + 1, y + 1);

                h_energy += gx.abs();
                v_energy += gy.abs();
                d1_energy += gd1.abs();
                d2_energy += gd2.abs();
            }
        }

        let total = h_energy + v_energy + d1_energy + d2_energy;
        let energies = [h_energy, v_energy, d1_energy, d2_energy];
        let max_e = energies.iter().cloned().fold(0.0f32, f32::max);
        let min_e = energies.iter().cloned().fold(f32::MAX, f32::min);
        let directional_bias = (max_e - min_e) / (total + 0.001);

        // Block variance analysis
        let block_size = 16u32;
        let blocks_x = self.width / block_size;
        let blocks_y = self.height / block_size;
        let mut block_variances = Vec::new();

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut sum = 0.0;
                let mut sq_sum = 0.0;
                let mut count = 0;

                for y in (by * block_size)..((by + 1) * block_size) {
                    for x in (bx * block_size)..((bx + 1) * block_size) {
                        let val = noise[(y * self.width + x) as usize];
                        sum += val;
                        sq_sum += val * val;
                        count += 1;
                    }
                }

                let mean = sum / count as f32;
                let variance = sq_sum / count as f32 - mean * mean;
                block_variances.push(variance);
            }
        }

        let mean_var: f32 = block_variances.iter().sum::<f32>() / block_variances.len() as f32;
        let var_of_var: f32 = block_variances.iter()
            .map(|v| (v - mean_var).powi(2))
            .sum::<f32>() / block_variances.len() as f32;
        let var_of_var = var_of_var.sqrt();

        let uniformity_score = 1.0 / (1.0 + var_of_var / 10.0);
        
        // uniformity > 0.65 = AI, < 0.50 = human
        let uniformity_prob = Self::to_probability(uniformity_score, 0.57, 10.0, false);
        // directional_bias < 0.10 = AI (no brush direction), > 0.15 = human
        let direction_prob = Self::to_probability(directional_bias, 0.12, 30.0, true);
        // Combine: high uniformity AND low direction = AI
        let ai_probability = (uniformity_prob * 0.5 + direction_prob * 0.5).clamp(0.0, 1.0);

        let max_noise = noise.iter().cloned().fold(0.0f32, f32::max);
        let data: Vec<u8> = noise.iter()
            .map(|&v| ((v / max_noise) * 255.0 * 2.0).min(255.0) as u8)
            .collect();

        MethodResult {
            data,
            ai_probability,
            metrics: vec![
                ("uniformity".into(), uniformity_score),
                ("directional_bias".into(), directional_bias),
            ],
        }
    }

    // Method 8: Impossible Gradients
    fn analyze_gradients(&self) -> MethodResult {
        let mut gradient_map = vec![0.0f32; (self.width * self.height) as usize];
        let window = 7i32;
        let half = window / 2;

        let mut perfect_count = 0u32;
        let mut total_count = 0u32;

        for y in (half..(self.height as i32 - half)).step_by(2) {
            for x in (half..(self.width as i32 - half)).step_by(2) {
                // Collect window values
                let mut values = Vec::new();
                for dy in -half..=half {
                    for dx in -half..=half {
                        values.push((dx as f32, dy as f32, self.get_gray(x + dx, y + dy)));
                    }
                }

                // Fit plane: v = ax + by + c using least squares
                let n = values.len() as f32;
                let (mut sx, mut sy, mut sv) = (0.0, 0.0, 0.0);
                let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
                let (mut sxv, mut syv) = (0.0, 0.0);

                for &(px, py, pv) in &values {
                    sx += px; sy += py; sv += pv;
                    sxx += px * px; syy += py * py; sxy += px * py;
                    sxv += px * pv; syv += py * pv;
                }

                let det = n * (sxx * syy - sxy * sxy) -
                         sx * (sx * syy - sy * sxy) +
                         sy * (sx * sxy - sy * sxx);

                if det.abs() > 0.001 {
                    let a = (sxv * (n * syy - sy * sy) -
                            syv * (n * sxy - sx * sy) +
                            sv * (sx * sy - sy * sxy)) / det;

                    let b = (sxx * (n * syv - sy * sv) -
                            sxy * (n * sxv - sx * sv) +
                            sx * (sxv * sy - syv * sx)) / det;

                    let c = sv / n - a * sx / n - b * sy / n;

                    // Calculate residual
                    let mut residual_sum = 0.0;
                    for &(px, py, pv) in &values {
                        let predicted = a * px + b * py + c;
                        residual_sum += (pv - predicted).abs();
                    }
                    let avg_residual = residual_sum / n;

                    let gradient_mag = (a * a + b * b).sqrt();
                    let has_gradient = gradient_mag > 0.5;
                    let is_perfect = avg_residual < 2.0;

                    if has_gradient {
                        total_count += 1;
                        if is_perfect {
                            perfect_count += 1;
                            gradient_map[(y as u32 * self.width + x as u32) as usize] = 
                                255.0 - avg_residual * 50.0;
                        } else {
                            gradient_map[(y as u32 * self.width + x as u32) as usize] = 
                                (128.0 - avg_residual * 10.0).max(0.0);
                        }
                    }
                }
            }
        }

        // Expand visualization
        let mut expanded = vec![0u8; (self.width * self.height) as usize];
        for y in 0..self.height {
            for x in 0..self.width {
                let ny = ((y / 2) * 2).min(self.height - 1);
                let nx = ((x / 2) * 2).min(self.width - 1);
                expanded[(y * self.width + x) as usize] = 
                    gradient_map[(ny * self.width + nx) as usize].min(255.0).max(0.0) as u8;
            }
        }

        let perfect_ratio = if total_count > 0 {
            perfect_count as f32 / total_count as f32
        } else {
            0.0
        };

        // perfect_ratio > 0.30 = AI, < 0.20 = human
        let ai_probability = Self::to_probability(perfect_ratio, 0.25, 20.0, false);

        MethodResult {
            data: expanded,
            ai_probability,
            metrics: vec![
                ("perfect_ratio".into(), perfect_ratio),
                ("perfect_count".into(), perfect_count as f32),
                ("total_gradients".into(), total_count as f32),
            ],
        }
    }

    // Method 9: CLIP-Space / Style Coherence
    fn analyze_clip_space(&self) -> MethodResult {
        let block_size = 64u32;
        let blocks_x = self.width / block_size;
        let blocks_y = self.height / block_size;

        if blocks_x < 2 || blocks_y < 2 {
            return MethodResult {
                data: vec![0; (self.width * self.height) as usize],
                ai_probability: 0.5, // Inconclusive for small images
                metrics: vec![
                    ("coherence".into(), 0.0),
                    ("uniformity".into(), 0.0),
                ],
            };
        }

        // Extract features for each block
        let mut block_features: Vec<Vec<f32>> = Vec::new();

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let feature = self.extract_block_features(
                    bx * block_size, by * block_size, block_size
                );
                block_features.push(feature);
            }
        }

        // Calculate coherence (similarity between all blocks)
        let mut total_sim = 0.0;
        let mut pair_count = 0;

        for i in 0..block_features.len() {
            for j in (i + 1)..block_features.len() {
                let sim = Self::cosine_similarity(&block_features[i], &block_features[j]);
                total_sim += sim;
                pair_count += 1;
            }
        }

        let coherence = if pair_count > 0 {
            total_sim / pair_count as f32
        } else {
            0.0
        };

        // Calculate mean feature
        let feat_len = block_features[0].len();
        let mut mean_feature = vec![0.0; feat_len];
        for feat in &block_features {
            for (i, &v) in feat.iter().enumerate() {
                mean_feature[i] += v / block_features.len() as f32;
            }
        }

        // Feature variance
        let mut feat_var = 0.0;
        for feat in &block_features {
            for (i, &v) in feat.iter().enumerate() {
                feat_var += (v - mean_feature[i]).powi(2);
            }
        }
        feat_var = (feat_var / (block_features.len() * feat_len) as f32).sqrt();

        let style_uniformity = 1.0 / (1.0 + feat_var);

        // Visualization
        let mut vis = vec![0.0f32; (self.width * self.height) as usize];
        for (idx, feat) in block_features.iter().enumerate() {
            let bx = idx as u32 % blocks_x;
            let by = idx as u32 / blocks_x;

            let mut deviation = 0.0;
            for (i, &v) in feat.iter().enumerate() {
                deviation += (v - mean_feature[i]).abs();
            }
            deviation /= feat_len as f32;

            for y in (by * block_size)..((by + 1) * block_size).min(self.height) {
                for x in (bx * block_size)..((bx + 1) * block_size).min(self.width) {
                    vis[(y * self.width + x) as usize] = 255.0 * (1.0 - deviation * 5.0);
                }
            }
        }

        let data: Vec<u8> = vis.iter()
            .map(|&v| v.min(255.0).max(0.0) as u8)
            .collect();

        // coherence > 0.80 = AI, < 0.70 = human
        let coherence_prob = Self::to_probability(coherence, 0.75, 15.0, false);
        // style_uniformity > 0.55 = AI, < 0.45 = human
        let uniformity_prob = Self::to_probability(style_uniformity, 0.50, 12.0, false);
        let ai_probability = (coherence_prob * 0.6 + uniformity_prob * 0.4).clamp(0.0, 1.0);

        MethodResult {
            data,
            ai_probability,
            metrics: vec![
                ("coherence".into(), coherence),
                ("uniformity".into(), style_uniformity),
            ],
        }
    }

    fn extract_block_features(&self, start_x: u32, start_y: u32, size: u32) -> Vec<f32> {
        let mut features = vec![0.0; 40];
        let mut color_hist = vec![0.0f32; 24];
        let mut pixel_count = 0u32;

        let end_x = (start_x + size).min(self.width);
        let end_y = (start_y + size).min(self.height);

        // Color histogram
        for y in start_y..end_y {
            for x in start_x..end_x {
                let idx = (y * self.width + x) as usize * 4;
                let r = self.data[idx] as usize;
                let g = self.data[idx + 1] as usize;
                let b = self.data[idx + 2] as usize;

                color_hist[r / 32] += 1.0;
                color_hist[8 + g / 32] += 1.0;
                color_hist[16 + b / 32] += 1.0;
                pixel_count += 1;
            }
        }

        for i in 0..24 {
            features[i] = color_hist[i] / pixel_count as f32;
        }

        // Edge histogram (8 directions)
        let mut edge_hist = vec![0.0f32; 8];
        for y in (start_y + 1)..(end_y - 1) {
            for x in (start_x + 1)..(end_x - 1) {
                let idx = (y * self.width + x) as usize * 4;
                let gray = 0.299 * self.data[idx] as f32 +
                          0.587 * self.data[idx + 1] as f32 +
                          0.114 * self.data[idx + 2] as f32;

                let idx_r = idx + 4;
                let idx_d = idx + (self.width as usize * 4);

                let gray_r = 0.299 * self.data[idx_r] as f32 +
                            0.587 * self.data[idx_r + 1] as f32 +
                            0.114 * self.data[idx_r + 2] as f32;

                let gray_d = 0.299 * self.data[idx_d] as f32 +
                            0.587 * self.data[idx_d + 1] as f32 +
                            0.114 * self.data[idx_d + 2] as f32;

                let gx = gray_r - gray;
                let gy = gray_d - gray;
                let mag = (gx * gx + gy * gy).sqrt();

                if mag > 5.0 {
                    let angle = gy.atan2(gx);
                    let bin = (((angle + PI) / (2.0 * PI)) * 8.0) as usize % 8;
                    edge_hist[bin] += mag;
                }
            }
        }

        let edge_total: f32 = edge_hist.iter().sum();
        for i in 0..8 {
            features[24 + i] = if edge_total > 0.0 { edge_hist[i] / edge_total } else { 0.0 };
        }

        // Texture (simplified LBP)
        let mut tex_bins = vec![0.0f32; 8];
        for y in ((start_y + 1)..(end_y - 1)).step_by(2) {
            for x in ((start_x + 1)..(end_x - 1)).step_by(2) {
                let idx = (y * self.width + x) as usize * 4;
                let center = self.data[idx];

                let neighbors = [
                    self.data[idx - (self.width as usize * 4) - 4],
                    self.data[idx - (self.width as usize * 4)],
                    self.data[idx - (self.width as usize * 4) + 4],
                    self.data[idx + 4],
                    self.data[idx + (self.width as usize * 4) + 4],
                    self.data[idx + (self.width as usize * 4)],
                    self.data[idx + (self.width as usize * 4) - 4],
                    self.data[idx - 4],
                ];

                let mut pattern = 0u8;
                for (i, &n) in neighbors.iter().enumerate() {
                    if n >= center {
                        pattern |= 1 << i;
                    }
                }

                let mut transitions = 0;
                for i in 0..8 {
                    let curr = (pattern >> i) & 1;
                    let next = (pattern >> ((i + 1) % 8)) & 1;
                    if curr != next {
                        transitions += 1;
                    }
                }

                tex_bins[transitions.min(7) as usize] += 1.0;
            }
        }

        let tex_total: f32 = tex_bins.iter().sum();
        for i in 0..8 {
            features[32 + i] = if tex_total > 0.0 { tex_bins[i] / tex_total } else { 0.0 };
        }

        features
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        dot / (norm_a.sqrt() * norm_b.sqrt() + 0.0001)
    }

    fn get_verdict(avg_probability: f32) -> (String, String, String) {
        let pct = (avg_probability * 100.0).round() as u32;
        let confidence_str = format!("{}%", pct);
        
        if avg_probability >= 0.75 {
            ("LIKELY AI-GENERATED".into(), confidence_str, "high".into())
        } else if avg_probability >= 0.55 {
            ("PROBABLY AI".into(), confidence_str, "medium".into())
        } else if avg_probability >= 0.40 {
            ("INCONCLUSIVE".into(), confidence_str, "medium".into())
        } else if avg_probability >= 0.25 {
            ("PROBABLY HUMAN".into(), confidence_str, "low".into())
        } else {
            ("LIKELY HUMAN-MADE".into(), confidence_str, "low".into())
        }
    }

    #[wasm_bindgen]
    pub fn analyze(&self) -> JsValue {
        let noise = self.analyze_noise();
        let high_pass = self.analyze_high_pass();
        let ela = self.analyze_ela();
        let posterize = self.analyze_posterize();
        let channels = self.analyze_channels();
        let (fft, fft_size) = self.analyze_fft();
        let noise_patterns = self.analyze_noise_patterns();
        let gradients = self.analyze_gradients();
        let clip_space = self.analyze_clip_space();

        // Weighted average of all probabilities
        // FFT and noise get higher weight as they're more reliable
        let total_weight = 2.0 + 1.0 + 1.0 + 1.0 + 1.0 + 2.0 + 1.5 + 1.0 + 1.5;
        let weighted_sum = 
            noise.ai_probability * 2.0 +
            high_pass.ai_probability * 1.0 +
            ela.ai_probability * 1.0 +
            posterize.ai_probability * 1.0 +
            channels.ai_probability * 1.0 +
            fft.ai_probability * 2.0 +
            noise_patterns.ai_probability * 1.5 +
            gradients.ai_probability * 1.0 +
            clip_space.ai_probability * 1.5;
        
        let avg_probability = weighted_sum / total_weight;
        let score = (avg_probability * 10.0).round() as u32;

        let (verdict, confidence, verdict_class) = Self::get_verdict(avg_probability);

        let result = AnalysisResult {
            score,
            max_score: 10,
            verdict,
            confidence,
            verdict_class,
            noise,
            high_pass,
            ela,
            posterize,
            channels,
            fft,
            fft_size,
            noise_patterns,
            gradients,
            clip_space,
        };

        serde_wasm_bindgen::to_value(&result).unwrap()
    }
}
