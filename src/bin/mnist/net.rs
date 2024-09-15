use tch::{
    nn::{batch_norm2d, conv2d, linear, BatchNorm, Conv2D, ConvConfig, Linear, ModuleT, Path},
    Kind, Tensor,
};

#[derive(Debug)]
pub struct Net {
    pub conv1: Conv2D,
    pub bn1: BatchNorm,
    pub conv2: Conv2D,
    pub bn2: BatchNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl Net {
    pub fn new(vs: &Path) -> Self {
        let conv1 = conv2d(
            vs,
            1,
            32,
            3,
            ConvConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
        );
        let bn1 = batch_norm2d(vs, 32, Default::default());
        let conv2 = conv2d(
            vs,
            32,
            64,
            3,
            ConvConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
        );
        let bn2 = batch_norm2d(vs, 64, Default::default());
        let fc1 = linear(vs, 64 * 7 * 7, 1024, Default::default());
        let fc2 = linear(vs, 1024, 10, Default::default());
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            fc1,
            fc2,
        }
    }
}

impl ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.view([-1, 1, 28, 28]);
        let xs = xs.apply(&self.conv1);
        let xs = xs.apply_t(&self.bn1, train);
        let xs = xs.relu();
        let xs = xs.apply(&self.conv2);
        let xs = xs.apply_t(&self.bn2, train);
        let xs = xs.relu();
        let xs = xs.view([xs.size()[0], -1]);
        let xs = xs.apply(&self.fc1);
        let xs = xs.relu();
        let xs = xs.apply(&self.fc2);

        match train {
            true => xs,
            false => xs.softmax(-1, Kind::Float),
        }
    }
}
