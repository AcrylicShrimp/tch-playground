mod batch;
mod data;
mod loss;
mod net;
mod trainer;

use batch::make_batch;
use data::load_mnist_image_set;
use log::info;
use net::Net;
use tch::{
    nn::{Adam, VarStore},
    Device,
};
use trainer::{LearningRateScheduler, Trainer};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    env_logger::init();

    let device = if tch::Cuda::is_available() {
        info!("CUDA is available, using CUDA");
        Device::cuda_if_available()
    } else if tch::utils::has_mps() {
        info!("MPS is available, using MPS");
        Device::Mps
    } else {
        info!("no accelerator available, using CPU");
        Device::Cpu
    };

    info!("loading data...");
    let mut train_image_set = load_mnist_image_set(
        "src/bin/mnist/data/train-images.idx3-ubyte",
        "src/bin/mnist/data/train-labels.idx1-ubyte",
    )?;
    let test_image_set = load_mnist_image_set(
        "src/bin/mnist/data/t10k-images.idx3-ubyte",
        "src/bin/mnist/data/t10k-labels.idx1-ubyte",
    )?;
    let test_batch = make_batch(
        device,
        test_image_set.image_width,
        test_image_set.image_height,
        &test_image_set.images,
    );

    train_image_set.augment(0.15, 0.15);

    let vs = VarStore::new(device);
    let net = Net::new(&vs.root());
    let mut trainer = Trainer::new(
        &vs,
        net,
        Adam::default(),
        LearningRateScheduler::new(1e-3, 0.9),
    )?;

    let final_accuracy = trainer.train(20, 32, train_image_set, &test_batch).await?;

    info!("================================================");
    info!("final accuracy: {:.2}%", final_accuracy * 100.0);
    info!("================================================");

    Ok(())
}
