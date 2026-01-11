pub fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len2 <= 1e-20 {
        return [0.0, 0.0, 0.0];
    }
    let inv = len2.sqrt().recip();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
