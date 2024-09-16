use crate::{
    batch::{Batch, BatchGenerator},
    data::MnistImageSet,
    loss::loss,
    net::Net,
};
use log::info;
use tch::{
    nn::{ModuleT, Optimizer, OptimizerConfig, VarStore},
    Device, Kind,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrainerError {
    #[error("tch error: {0:?}")]
    Tch(#[from] tch::TchError),
    #[error("batch generator error: {0:?}")]
    BatchGenerator(#[from] crate::batch::BatchGeneratorError),
    #[error("{0:?}")]
    Anyhow(#[from] anyhow::Error),
}

#[derive(Debug)]
pub struct Trainer {
    device: Device,
    net: Net,
    optimizer: Optimizer,
    lr_scheduler: LearningRateScheduler,
}

impl Trainer {
    pub fn new(
        vs: &VarStore,
        net: Net,
        optimizer_config: impl OptimizerConfig,
        lr_scheduler: LearningRateScheduler,
    ) -> Result<Self, TrainerError> {
        let optimizer = optimizer_config.build(vs, lr_scheduler.lr())?;

        Ok(Self {
            device: vs.device(),
            net,
            optimizer,
            lr_scheduler,
        })
    }

    pub async fn train(
        &mut self,
        epochs: usize,
        batch_size: usize,
        train_image_set: MnistImageSet,
        test_batch: &Batch,
    ) -> Result<f32, TrainerError> {
        let mut batch_generator = BatchGenerator::new(self.device, batch_size, train_image_set)?;

        for epoch in 0..epochs {
            info!("============= epoch {}/{} =============", epoch + 1, epochs);

            let test_accuracy = compute_test_accuracy(&self.net, test_batch)?;
            info!("test accuracy: {:.2}%", test_accuracy * 100.0);

            loop {
                let batch = batch_generator.next().await?;
                let batch = match batch {
                    Some(batch) => batch,
                    None => break,
                };
                let logits = self.net.forward_t(&batch.images, true);
                let loss = loss(&logits, &batch.labels);
                self.optimizer.backward_step(&loss);
            }

            self.lr_scheduler.step();
            self.optimizer.set_lr(self.lr_scheduler.lr());

            info!("reduced learning rate: {}", self.lr_scheduler.lr());
        }

        let test_accuracy = compute_test_accuracy(&self.net, test_batch)?;
        info!("test accuracy: {:.2}%", test_accuracy * 100.0);

        Ok(test_accuracy)
    }
}

fn compute_test_accuracy(net: &Net, test_batch: &Batch) -> Result<f32, anyhow::Error> {
    let logits = net.forward_t(&test_batch.images, false);
    let labels = logits.argmax(-1, false);

    let accuracy = labels.eq_tensor(&test_batch.labels).mean(Kind::Float);
    let accuracy = f32::try_from(accuracy)?;

    Ok(accuracy)
}

#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    lr: f64,
    lr_decay: f64,
}

impl LearningRateScheduler {
    pub fn new(lr: f64, lr_decay: f64) -> Self {
        Self { lr, lr_decay }
    }

    pub fn lr(&self) -> f64 {
        self.lr
    }

    pub fn step(&mut self) {
        self.lr *= self.lr_decay;
    }
}
