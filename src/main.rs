use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use anyhow::{anyhow, Result};
use clap::Parser;
use ndarray::{Array0, Array1, Array2};
use ort::inputs;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[derive(Parser, Debug)]
#[command(author, version, about = "Rust CLI for Generative Adaptive MIDI Extractor (GAME) ONNX inference.")]
struct Args {
    /// Path to input audio file (WAV, FLAC, MP3, etc.)
    input: PathBuf,

    /// Path to output MIDI file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to the directory containing ONNX models (config.json, encoder.onnx, etc.)
    #[arg(short, long, default_value = "checkpoints/GAME-1.0-large-onnx")]
    model_dir: PathBuf,

    /// Tempo (BPM) for output MIDI
    #[arg(long, default_value_t = 120.0)]
    tempo: f32,

    /// Boundary decoding threshold for segmentation model
    #[arg(long, default_value_t = 0.2)]
    seg_threshold: f32,

    /// Boundary decoding radius for local maxima search
    #[arg(long, default_value_t = 2)]
    seg_radius: i64,

    /// Presence detecting threshold for estimation model
    #[arg(long, default_value_t = 0.2)]
    est_threshold: f32,

    /// Language code (e.g. en, ja, zh) to improve segmentation
    #[arg(short, long)]
    language: Option<String>,

    /// Number of D3PM sampling steps (nsteps)
    #[arg(long, default_value_t = 8)]
    nsteps: usize,

    /// Starting T value (t0) for D3PM
    #[arg(long, default_value_t = 0.0)]
    t0: f32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ModelConfig {
    samplerate: u32,
    timestep: f32,
    languages: Option<HashMap<String, i64>>,
    loop_enabled: Option<bool>,
    embedding_dim: usize,
}

#[derive(Clone, Debug)]
struct NoteInfo {
    onset: f32,
    offset: f32,
    pitch: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ort::init().commit();

    // 1. Load configuration
    let config_path = args.model_dir.join("config.json");
    if !config_path.exists() {
        return Err(anyhow!("Model config file not found: {:?}", config_path));
    }
    let mut config_file = File::open(&config_path)?;
    let mut config_data = String::new();
    config_file.read_to_string(&mut config_data)?;
    let config: ModelConfig = serde_json::from_str(&config_data)?;
    println!("Loaded model configuration: {:?}", config);

    // 2. Load and resample audio
    println!("Loading audio file: {:?}", args.input);
    let (raw_samples, orig_sr) = load_audio(&args.input)?;
    println!(
        "Loaded {} samples at {} Hz (mono)",
        raw_samples.len(),
        orig_sr
    );

    let resampled = resample(&raw_samples, orig_sr, config.samplerate);
    println!(
        "Resampled to {} samples at {} Hz",
        resampled.len(),
        config.samplerate
    );

    let total_samples = resampled.len();
    let audio_duration = total_samples as f32 / config.samplerate as f32;
    println!("Audio duration: {:.2} seconds", audio_duration);

    // 3. Resolve language ID
    let language_id = if let Some(lang_code) = &args.language {
        if let Some(ref lang_map) = config.languages {
            *lang_map.get(lang_code).unwrap_or(&0)
        } else {
            0
        }
    } else {
        0
    };
    println!("Selected language: {:?} -> ID: {}", args.language, language_id);

    // 4. Initialize ONNX Sessions
    println!("Initializing ONNX sessions from {:?}", args.model_dir);
    let mut encoder_sess = Session::builder()?.commit_from_file(args.model_dir.join("encoder.onnx"))?;
    let mut segmenter_sess = Session::builder()?.commit_from_file(args.model_dir.join("segmenter.onnx"))?;
    let mut bd2dur_sess = Session::builder()?.commit_from_file(args.model_dir.join("bd2dur.onnx"))?;
    let mut estimator_sess = Session::builder()?.commit_from_file(args.model_dir.join("estimator.onnx"))?;
    println!("All ONNX sessions successfully initialized!");

    // 5. Run Encoder
    let waveform_arr = Array2::from_shape_vec((1, total_samples), resampled)?;
    let duration_arr = Array1::from_vec(vec![audio_duration]);

    let waveform_val = Tensor::from_array(waveform_arr)?;
    let duration_val = Tensor::from_array(duration_arr)?;

    println!("Running encoder...");
    let encoder_outputs = encoder_sess.run(inputs![
        "waveform" => waveform_val,
        "duration" => duration_val,
    ])?;

    let (x_seg_shape, x_seg_data) = encoder_outputs[0].try_extract_tensor::<f32>()?;
    let (x_est_shape, x_est_data) = encoder_outputs[1].try_extract_tensor::<f32>()?;
    let (mask_t_shape, mask_t_data) = encoder_outputs[2].try_extract_tensor::<bool>()?;

    let num_frames = mask_t_shape[1] as usize;
    println!("Encoder outputs shape - x_seg: {:?}, mask_t: {:?}", x_seg_shape, mask_t_shape);

    let x_seg_tensor = Tensor::from_array((
        x_seg_shape.to_vec(),
        x_seg_data.to_vec().into_boxed_slice()
    ))?;
    let x_est_tensor = Tensor::from_array((
        x_est_shape.to_vec(),
        x_est_data.to_vec().into_boxed_slice()
    ))?;
    let mask_t_tensor = Tensor::from_array((
        mask_t_shape.to_vec(),
        mask_t_data.to_vec().into_boxed_slice()
    ))?;

    // 6. Run Segmentation Loop (D3PM)
    let step = (1.0 - args.t0) / args.nsteps as f32;
    let ts: Vec<f32> = (0..args.nsteps)
        .map(|i| args.t0 + i as f32 * step)
        .collect();

    let mut boundaries_val = Tensor::from_array(Array2::<bool>::from_elem((1, num_frames), false))?;
    let known_boundaries_val = Tensor::from_array(Array2::<bool>::from_elem((1, num_frames), false))?;

    let language_val = Tensor::from_array(Array1::from_vec(vec![language_id]))?;
    let seg_threshold_val = Tensor::from_array(Array0::from_elem((), args.seg_threshold))?;
    let seg_radius_val = Tensor::from_array(Array0::from_elem((), args.seg_radius))?;

    println!("Running segmenter loop (D3PM, {} steps)...", ts.len());
    for &t_val in &ts {
        let t_tensor = Tensor::from_array(Array1::from_vec(vec![t_val]))?;
        let outputs = segmenter_sess.run(inputs![
            "x_seg" => &x_seg_tensor,
            "language" => &language_val,
            "known_boundaries" => &known_boundaries_val,
            "prev_boundaries" => &boundaries_val,
            "t" => &t_tensor,
            "maskT" => &mask_t_tensor,
            "threshold" => &seg_threshold_val,
            "radius" => &seg_radius_val,
        ])?;
        let (boundaries_shape, boundaries_data) = outputs[0].try_extract_tensor::<bool>()?;
        boundaries_val = Tensor::from_array((
            boundaries_shape.to_vec(),
            boundaries_data.to_vec().into_boxed_slice()
        ))?;
    }

    // 7. Convert boundaries to durations
    println!("Converting boundaries to durations...");
    let bd2dur_outputs = bd2dur_sess.run(inputs![
        "boundaries" => &boundaries_val,
        "maskT" => &mask_t_tensor,
    ])?;

    let (durations_shape, durations_data) = bd2dur_outputs[0].try_extract_tensor::<f32>()?;
    let (mask_n_shape, mask_n_data) = bd2dur_outputs[1].try_extract_tensor::<bool>()?;

    println!("Durations shape: {:?}, Note mask shape: {:?}", durations_shape, mask_n_shape);

    let _durations_tensor = Tensor::from_array((
        durations_shape.to_vec(),
        durations_data.to_vec().into_boxed_slice()
    ))?;
    let mask_n_tensor = Tensor::from_array((
        mask_n_shape.to_vec(),
        mask_n_data.to_vec().into_boxed_slice()
    ))?;

    // 8. Run Estimator
    let est_threshold_val = Tensor::from_array(Array0::from_elem((), args.est_threshold))?;

    println!("Running pitch estimator...");
    let estimator_outputs = estimator_sess.run(inputs![
        "x_est" => &x_est_tensor,
        "boundaries" => &boundaries_val,
        "maskT" => &mask_t_tensor,
        "maskN" => &mask_n_tensor,
        "threshold" => &est_threshold_val,
    ])?;

    let (presence_shape, presence_data) = estimator_outputs[0].try_extract_tensor::<bool>()?;
    let (scores_shape, scores_data) = estimator_outputs[1].try_extract_tensor::<f32>()?;

    println!("Presence shape: {:?}, Scores shape: {:?}", presence_shape, scores_shape);

    // 9. Decode notes and durations
    let mut notes = Vec::new();
    let mut current_time = 0.0;

    for i in 0..presence_data.len() {
        let dur = durations_data[i];
        let onset = current_time;
        let offset = current_time + dur;
        current_time = offset;

        let valid = presence_data[i];
        let pitch = scores_data[i];

        if offset - onset <= 0.0 {
            continue;
        }
        if !valid {
            continue;
        }

        notes.push(NoteInfo {
            onset,
            offset,
            pitch,
        });
    }

    // Sort notes by onset, offset, pitch
    notes.sort_by(|a, b| {
        a.onset.partial_cmp(&b.onset).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.offset.partial_cmp(&b.offset).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.pitch.partial_cmp(&b.pitch).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Clean up overlapping notes
    let mut clean_notes = Vec::new();
    let mut last_time = 0.0;
    for mut note in notes {
        note.onset = note.onset.max(last_time);
        note.offset = note.offset.max(note.onset);
        if note.offset > note.onset {
            last_time = note.offset;
            clean_notes.push(note);
        }
    }

    println!("Detected {} voiced notes.", clean_notes.len());

    // 10. Write MIDI file
    let output_path = args.output.unwrap_or_else(|| {
        let mut path = args.input.clone();
        path.set_extension("mid");
        path
    });

    println!("Writing MIDI file to {:?}", output_path);
    write_midi(&output_path, &clean_notes, args.tempo)?;
    println!("Successfully saved MIDI to {:?}", output_path);

    Ok(())
}

fn load_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
    let src = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no audio track found"))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);

    let dec_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(ref err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(err) => return Err(err.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let num_channels = decoded.spec().channels.count();
                let num_frames = decoded.frames();

                let spec = *decoded.spec();
                let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);
                let interleaved_samples = sample_buf.samples();

                for f in 0..num_frames {
                    let mut sum = 0.0;
                    for c in 0..num_channels {
                        sum += interleaved_samples[f * num_channels + c];
                    }
                    samples.push(sum / num_channels as f32);
                }
            }
            Err(SymphoniaError::DecodeError(err)) => {
                eprintln!("decode error: {}", err);
            }
            Err(err) => return Err(err.into()),
        }
    }

    Ok((samples, sample_rate))
}

fn resample(samples: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == to_sr {
        return samples.to_vec();
    }
    let ratio = to_sr as f64 / from_sr as f64;
    let new_len = (samples.len() as f64 * ratio).round() as usize;
    let mut resampled = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let pos = i as f64 / ratio;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        if idx + 1 < samples.len() {
            let s0 = samples[idx];
            let s1 = samples[idx + 1];
            resampled.push(s0 + frac as f32 * (s1 - s0));
        } else if idx < samples.len() {
            resampled.push(samples[idx]);
        }
    }
    resampled
}

fn write_midi(path: &Path, notes: &[NoteInfo], tempo_bpm: f32) -> Result<()> {
    use midly::{Header, Smf, Track, TrackEvent, TrackEventKind, MidiMessage, Timing, Format};
    use midly::num::u24;

    let header = Header::new(Format::SingleTrack, Timing::Metrical(480.into()));
    let mut track = Track::new();

    // Set tempo in track
    let mpb = (60_000_000.0 / tempo_bpm).round() as u32;
    track.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(midly::MetaMessage::Tempo(u24::from_int_lossy(mpb))),
    });

    let mut last_time_ticks = 0u32;
    for note in notes {
        let onset_ticks = (note.onset * tempo_bpm * 8.0).round() as u32;
        let offset_ticks = (note.offset * tempo_bpm * 8.0).round() as u32;
        let midi_pitch = note.pitch.round().clamp(0.0, 127.0) as u8;

        if offset_ticks <= onset_ticks {
            continue;
        }

        // Note On event
        let delta_on = onset_ticks.checked_sub(last_time_ticks).unwrap_or(0);
        track.push(TrackEvent {
            delta: delta_on.into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOn {
                    key: midi_pitch.into(),
                    vel: 127.into(),
                },
            },
        });

        // Note Off event
        let delta_off = offset_ticks - onset_ticks;
        track.push(TrackEvent {
            delta: delta_off.into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOff {
                    key: midi_pitch.into(),
                    vel: 0.into(),
                },
            },
        });

        last_time_ticks = offset_ticks;
    }

    let smf = Smf {
        header,
        tracks: vec![track],
    };

    smf.save(path)?;
    Ok(())
}
