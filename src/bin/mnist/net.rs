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
    pub conv3: Conv2D,
    pub bn3: BatchNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl Net {
    pub fn new(vs: &Path) -> Self {
        let conv1 = conv2d(
            vs,
            1,
            16,
            3,
            ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let bn1 = batch_norm2d(vs, 16, Default::default());
        let conv2 = conv2d(
            vs,
            16,
            32,
            3,
            ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let bn2 = batch_norm2d(vs, 32, Default::default());
        let conv3 = conv2d(
            vs,
            32,
            64,
            3,
            ConvConfig {
                padding: 0,
                ..Default::default()
            },
        );
        let bn3 = batch_norm2d(vs, 64, Default::default());
        let fc1 = linear(vs, 64 * 5 * 5, 512, Default::default());
        let fc2 = linear(vs, 512, 10, Default::default());
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
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
        // 28x28 -> 14x14
        let xs = xs.max_pool2d_default(2);
        let xs = xs.apply(&self.conv2);
        let xs = xs.apply_t(&self.bn2, train);
        let xs = xs.relu();
        // 14x14 -> 7x7
        let xs = xs.max_pool2d_default(2);
        // 7x7 -> 5x5
        let xs = xs.apply(&self.conv3);
        let xs = xs.apply_t(&self.bn3, train);
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
