use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub struct AudioCapture {
    _stream: cpal::Stream,
    pub sample_rate: u32,
}

pub fn list_input_devices() -> Result<Vec<(String, String)>> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    for device in host.input_devices()? {
        let name = device.to_string();
        let id = device
            .id()
            .map(|d| format!("{:?}", d))
            .unwrap_or_default();
        devices.push((name, id));
    }

    Ok(devices)
}

pub fn start_capture<F>(device_name: Option<&str>, mut callback: F) -> Result<AudioCapture>
where
    F: FnMut(&[f32]) + Send + 'static,
{
    let host = cpal::default_host();
    let device = match device_name {
        Some(name) => host
            .input_devices()?
            .find(|d| d.to_string() == name)
            .ok_or_else(|| anyhow::anyhow!("Device not found: {}", name))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device"))?,
    };

    let supported = device.default_input_config()?;
    let config = supported.config();
    let sample_format = supported.sample_format();
    let sample_rate = supported.sample_rate();
    let channels = config.channels as usize;

    let err_fn = |err| eprintln!("Audio stream error: {:?}", err);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if data.is_empty() {
                    return;
                }
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| {
                        let sum: f32 = frame.iter().copied().sum();
                        sum / channels as f32
                    })
                    .collect();
                callback(&mono);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                if data.is_empty() {
                    return;
                }
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| {
                        let sum: f32 = frame
                            .iter()
                            .map(|&s| s as f32 / i16::MAX as f32)
                            .sum();
                        sum / channels as f32
                    })
                    .collect();
                callback(&mono);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                if data.is_empty() {
                    return;
                }
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| {
                        let sum: f32 = frame
                            .iter()
                            .map(|&s| {
                                let centered = s as f32 - 32768.0;
                                centered / 32768.0
                            })
                            .sum();
                        sum / channels as f32
                    })
                    .collect();
                callback(&mono);
            },
            err_fn,
            None,
        )?,
        other => return Err(anyhow::anyhow!("Unsupported sample format: {:?}", other)),
    };

    stream.play()?;

    Ok(AudioCapture {
        _stream: stream,
        sample_rate,
    })
}
