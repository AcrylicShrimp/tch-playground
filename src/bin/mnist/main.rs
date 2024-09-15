mod batch;
mod data;
mod loss;
mod net;

use batch::{make_batch, Batch, BatchGenerator};
use data::{load_mnist_image_set, MnistImageSet};
use log::info;
use loss::loss;
use net::Net;
use tch::{
    nn::{Adam, ModuleT, Optimizer, OptimizerConfig, VarStore},
    Device, Kind,
};

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
    let train_image_set = load_mnist_image_set(
        "src/bin/mnist/data/train-images.idx3-ubyte",
        "src/bin/mnist/data/train-labels.idx1-ubyte",
    )?;
    let test_image_set = load_mnist_image_set(
        "src/bin/mnist/data/t10k-images.idx3-ubyte",
        "src/bin/mnist/data/t10k-labels.idx1-ubyte",
    )?;
    let test_batch = make_batch(
        device,
        train_image_set.image_width,
        train_image_set.image_height,
        &test_image_set.images,
    );

    let vs = VarStore::new(device);
    let net = Net::new(&vs.root());
    let optimizer = Adam::default().build(&vs, 1e-3)?;

    train(
        device,
        10,
        32,
        train_image_set,
        &test_batch,
        net,
        optimizer,
        1e-3,
        0.95,
    )
    .await?;

    Ok(())
}

async fn train(
    device: Device,
    epochs: usize,
    batch_size: usize,
    train_image_set: MnistImageSet,
    test_batch: &Batch,
    net: Net,
    mut optimizer: Optimizer,
    mut lr: f64,
    lr_decay: f64,
) -> Result<(), anyhow::Error> {
    let mut batch_generator = BatchGenerator::new(device, batch_size, train_image_set)?;
    optimizer.set_lr(lr);

    for epoch in 0..epochs {
        info!("============= epoch {}/{} =============", epoch + 1, epochs);

        let test_accuracy = compute_test_accuracy(&net, test_batch)?;
        info!("test accuracy: {:.2}%", test_accuracy * 100.0);

        loop {
            let batch = batch_generator.next().await?;
            let batch = match batch {
                Some(batch) => batch,
                None => break,
            };
            let logits = net.forward_t(&batch.images, true);
            let loss = loss(&logits, &batch.labels);
            optimizer.backward_step(&loss);
        }

        lr *= lr_decay;
        optimizer.set_lr(lr);

        info!("reduced learning rate: {}", lr);
    }

    let test_accuracy = compute_test_accuracy(&net, test_batch)?;
    info!("test accuracy: {:.2}%", test_accuracy * 100.0);

    Ok(())
}

fn compute_test_accuracy(net: &Net, test_batch: &Batch) -> Result<f32, anyhow::Error> {
    let logits = net.forward_t(&test_batch.images, false);
    let labels = logits.argmax(-1, false);

    let accuracy = labels.eq_tensor(&test_batch.labels).mean(Kind::Float);
    let accuracy = f32::try_from(accuracy)?;

    Ok(accuracy)
}
