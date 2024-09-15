use crate::data::{MnistImage, MnistImageSet};
use std::time::Duration;
use tch::{Device, Tensor};
use thiserror::Error;
use tokio::sync::mpsc::{Receiver, Sender};

#[derive(Error, Debug)]
pub enum BatchGeneratorError {
    #[error("invalid batch size: {0}")]
    InvalidBatchSize(usize),
    #[error("image set is empty")]
    EmptyImageSet,
    #[error("channel closed unexpectedly")]
    ChannelClosed,
}

#[derive(Debug)]
pub struct BatchGenerator {
    pub rx: Receiver<Option<Batch>>,
    pub tx_stop: Option<tokio::sync::oneshot::Sender<()>>,
}

#[derive(Debug)]
pub struct Batch {
    pub size: usize,
    pub images: Tensor,
    pub labels: Tensor,
}

impl BatchGenerator {
    pub fn new(
        device: Device,
        batch_size: usize,
        image_set: MnistImageSet,
    ) -> Result<Self, BatchGeneratorError> {
        if batch_size == 0 {
            return Err(BatchGeneratorError::InvalidBatchSize(batch_size));
        }

        if image_set.images.is_empty() {
            return Err(BatchGeneratorError::EmptyImageSet);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let (tx_stop, rx_stop) = tokio::sync::oneshot::channel();

        tokio::spawn(async move {
            batch_generator_loop(device, batch_size, image_set, tx, rx_stop).await;
        });

        Ok(Self {
            rx,
            tx_stop: Some(tx_stop),
        })
    }

    /// Returns the next batch in the generator.
    ///
    /// It returns `None` if it reaches the end of the image set (an epoch is fully completed).
    /// But it doesn't mean that it is finished. You can continue to call this method for next epoch.
    pub async fn next(&mut self) -> Result<Option<Batch>, BatchGeneratorError> {
        self.rx
            .recv()
            .await
            .ok_or(BatchGeneratorError::ChannelClosed)
    }
}

impl Drop for BatchGenerator {
    fn drop(&mut self) {
        if let Some(tx) = self.tx_stop.take() {
            let _ = tx.send(());
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

async fn batch_generator_loop(
    device: Device,
    batch_size: usize,
    mut image_set: MnistImageSet,
    tx: Sender<Option<Batch>>,
    mut rx_stop: tokio::sync::oneshot::Receiver<()>,
) {
    image_set.shuffle();
    let mut index = 0;

    loop {
        let batch = prepare_batch(device, batch_size, &mut index, &mut image_set);
        tokio::select! {
            result = tx.send(batch) => {
                if let Err(err) = result {
                    eprintln!("error sending batch: {err:#?}");
                    break;
                }
            }
            _ = &mut rx_stop => {
                break;
            }
        }
    }
}

fn prepare_batch(
    device: Device,
    batch_size: usize,
    index: &mut usize,
    image_set: &mut MnistImageSet,
) -> Option<Batch> {
    if image_set.images.len() <= *index {
        *index = 0;
        image_set.shuffle();
        return None;
    }

    let batch_size = batch_size.min(image_set.images.len() - *index);
    let batch = make_batch(
        device,
        image_set.image_width,
        image_set.image_height,
        &image_set.images[*index..*index + batch_size],
    );
    *index += batch_size;

    Some(batch)
}

pub fn make_batch(
    device: Device,
    image_width: u32,
    image_height: u32,
    images: &[MnistImage],
) -> Batch {
    let mut image_data =
        Vec::<f32>::with_capacity(images.len() * (image_width * image_height) as usize);
    let mut label_data = Vec::with_capacity(images.len());

    for image in images {
        image_data.extend(image.image.iter());
        label_data.push(image.label.to_index() as i32);
    }

    let image_tensor = Tensor::from_slice(&image_data)
        .reshape([
            images.len() as i64,
            1,
            image_width as i64,
            image_height as i64,
        ])
        .to_device(device);
    let label_tensor = Tensor::from_slice(&label_data)
        .reshape([images.len() as i64])
        .to_device(device);

    Batch {
        size: images.len(),
        images: image_tensor,
        labels: label_tensor,
    }
}
