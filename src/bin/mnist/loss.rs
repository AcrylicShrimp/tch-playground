use tch::Tensor;

pub fn loss(logits: &Tensor, labels: &Tensor) -> Tensor {
    logits.cross_entropy_for_logits(labels)
}
