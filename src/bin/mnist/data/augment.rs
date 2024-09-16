use super::MnistImage;
use rand::Rng;

pub fn augment_translate(
    image_width: u32,
    image_height: u32,
    images: &[MnistImage],
) -> Vec<MnistImage> {
    let mut rng = rand::thread_rng();
    let mut images = images.to_vec();

    for image in &mut images {
        let x_offset: i32 = rng.gen_range(-6..=6);
        let y_offset: i32 = rng.gen_range(-6..=6);
        let mut new_image = vec![0.0f32; image_width as usize * image_height as usize];

        for y in 0..image_height as i32 {
            for x in 0..image_width as i32 {
                let new_x = (x + x_offset) as u32 % image_width;
                let new_y = (y + y_offset) as u32 % image_height;
                new_image[new_y as usize * image_width as usize + new_x as usize] =
                    image.image[y as usize * image_width as usize + x as usize];
            }
        }

        image.image = new_image;
    }

    images
}

pub fn augment_erase_patch(
    image_width: u32,
    image_height: u32,
    images: &[MnistImage],
) -> Vec<MnistImage> {
    let mut rng = rand::thread_rng();
    let mut images = images.to_vec();

    for image in &mut images {
        let patch_x: i32 = rng.gen_range(8..=image_width as i32 - 8 - 4);
        let patch_y: i32 = rng.gen_range(8..=image_height as i32 - 8 - 4);
        let mut new_image = vec![0.0f32; image_width as usize * image_height as usize];

        for y in 0..image_height as i32 {
            for x in 0..image_width as i32 {
                new_image[y as usize * image_width as usize + x as usize] =
                    image.image[y as usize * image_width as usize + x as usize];

                if x >= patch_x && x < patch_x + 4 && y >= patch_y && y < patch_y + 4 {
                    new_image[y as usize * image_width as usize + x as usize] = 0.0;
                }
            }
        }

        image.image = new_image;
    }

    images
}
