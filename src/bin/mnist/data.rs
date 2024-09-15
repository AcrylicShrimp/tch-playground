use byteorder::{BigEndian, ReadBytesExt};
use rand::seq::SliceRandom;
use std::{fs::OpenOptions, io::Read, path::Path};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MnistImageLoadError {
    #[error("invalid image magic number: {0}")]
    InvalidImageMagicNumber(u32),
    #[error("invalid label magic number: {0}")]
    InvalidLabelMagicNumber(u32),
    #[error("image and label count mismatch: {0} != {1}")]
    CountMismatch(u32, u32),
    #[error("invalid label: {0}")]
    InvalidLabel(u8),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct MnistImageSet {
    pub image_width: u32,
    pub image_height: u32,
    pub images: Vec<MnistImage>,
}

impl MnistImageSet {
    pub fn shuffle(&mut self) {
        self.images.shuffle(&mut rand::thread_rng());
    }
}

#[derive(Debug, Clone)]
pub struct MnistImage {
    pub image: Vec<f32>,
    pub label: MnistImageLabel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MnistImageLabel {
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
}

impl MnistImageLabel {
    pub fn to_index(self) -> usize {
        match self {
            MnistImageLabel::Digit0 => 0,
            MnistImageLabel::Digit1 => 1,
            MnistImageLabel::Digit2 => 2,
            MnistImageLabel::Digit3 => 3,
            MnistImageLabel::Digit4 => 4,
            MnistImageLabel::Digit5 => 5,
            MnistImageLabel::Digit6 => 6,
            MnistImageLabel::Digit7 => 7,
            MnistImageLabel::Digit8 => 8,
            MnistImageLabel::Digit9 => 9,
        }
    }
}

pub fn load_mnist_image_set(
    image: impl AsRef<Path>,
    label: impl AsRef<Path>,
) -> Result<MnistImageSet, MnistImageLoadError> {
    let mut image_file = OpenOptions::new().read(true).open(image)?;
    let mut label_file = OpenOptions::new().read(true).open(label)?;

    let image_magic = image_file.read_u32::<BigEndian>()?;
    let image_count = image_file.read_u32::<BigEndian>()?;
    let image_width = image_file.read_u32::<BigEndian>()?;
    let image_height = image_file.read_u32::<BigEndian>()?;

    if image_magic != 0x00000803 {
        return Err(MnistImageLoadError::InvalidImageMagicNumber(image_magic));
    }

    let label_magic = label_file.read_u32::<BigEndian>()?;
    let label_count = label_file.read_u32::<BigEndian>()?;

    if label_magic != 0x00000801 {
        return Err(MnistImageLoadError::InvalidLabelMagicNumber(label_magic));
    }

    if image_count != label_count {
        return Err(MnistImageLoadError::CountMismatch(image_count, label_count));
    }

    let num = image_count as usize;
    let pixel_count = (image_width * image_height) as usize;
    let mut images = vec![0u8; pixel_count * num];
    let mut labels = vec![0u8; num];

    image_file.read_exact(&mut images)?;
    label_file.read_exact(&mut labels)?;

    let mut image_set = MnistImageSet {
        image_width,
        image_height,
        images: Vec::with_capacity(num),
    };

    for index in 0..num {
        let image = &images[index * pixel_count..(index + 1) * pixel_count];
        let label = labels[index];

        let image = MnistImage {
            image: image.iter().map(|x| *x as f32 / 255.0).collect(),
            label: match label {
                0 => MnistImageLabel::Digit0,
                1 => MnistImageLabel::Digit1,
                2 => MnistImageLabel::Digit2,
                3 => MnistImageLabel::Digit3,
                4 => MnistImageLabel::Digit4,
                5 => MnistImageLabel::Digit5,
                6 => MnistImageLabel::Digit6,
                7 => MnistImageLabel::Digit7,
                8 => MnistImageLabel::Digit8,
                9 => MnistImageLabel::Digit9,
                _ => {
                    return Err(MnistImageLoadError::InvalidLabel(label));
                }
            },
        };
        image_set.images.push(image);
    }

    Ok(image_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_mnist_image_set_train() {
        let image_set = load_mnist_image_set(
            "src/bin/mnist/data/train-images.idx3-ubyte",
            "src/bin/mnist/data/train-labels.idx1-ubyte",
        )
        .unwrap();

        assert_eq!(image_set.images.len(), 60000);
        assert_eq!(image_set.image_width, 28);
        assert_eq!(image_set.image_height, 28);
    }

    #[test]
    fn test_load_mnist_image_set_test() {
        let image_set = load_mnist_image_set(
            "src/bin/mnist/data/t10k-images.idx3-ubyte",
            "src/bin/mnist/data/t10k-labels.idx1-ubyte",
        )
        .unwrap();

        assert_eq!(image_set.images.len(), 10000);
        assert_eq!(image_set.image_width, 28);
        assert_eq!(image_set.image_height, 28);
    }
}
