original github: (https://github.com/CompVis/latent-diffusion)

create conda env: ``` conda env create -f environment.yaml ```

activate conda env: ``` conda activate ldm ``` 


[Notebook](latent_imagenet_diffusion_modified.ipynb)

I modified the orginal notebook from the github to deal with some of the old version imports. 

(Should be able to run in colab without error)

The notebook:

Demonstrates class-conditional image synthesis using Latent Diffusion Models (LDMs).
Instead of performing diffusion directly on high-resolution pixel images, the model operates in a latent space learned by a pretrained autoencoder (VAE/VQ-GAN). This makes sampling faster and more memory-efficient while preserving image quality.

During generation, the model starts from random Gaussian noise in latent space and iteratively denoises it using a trained U-Net.

- **Load pretrained LDM (`cin256-v2`)** – initialize the model and configuration.
- **Use DDIM sampling** – deterministic denoising in 20 steps for high-quality synthesis.
- **Apply Classifier-Free Guidance (CFG)**  
  - *Low guidance*  = more variety, less precision  
  - *High guidance* = sharper images, less diversity
- **Decode latent representations** – convert denoised latents back to RGB images via the VAE decoder.
- **Visualize outputs** – display multiple class samples in a grid.