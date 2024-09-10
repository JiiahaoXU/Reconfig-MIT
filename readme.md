## Data-Efficient Alignment in Medical Imaging via Reconfigurable Generative Networks, WACV 2025


### Abstract

---
Recent advances in deep learning have witnessed many successful medical image translation models that learn correspondences between two visual domains. However, building robust mappings between domains is a significant challenge when handling misalignments caused by factors such as respiratory motion and anatomical changes. This issue is further exacerbated in scenarios with limited data availability, leading to a significant degradation in translation quality. In this paper, we introduce a novel data-efficient framework for aligning medical images via Reconfigurable Generative Network (Reconfig-MIT) for high-quality image translation. The key idea of Reconfig-MIT is to adaptively expand the generative network width within a Generative Adversarial Networks (GAN) architecture, initially expanding rapidly to capture low-level features and then slowing to refine high-level complexities. This dynamic network adaptation mechanism allows to adaptively learn at different rates, thus the model can better respond to deviations in the data caused by misalignments, while maintaining an effective equilibrium with the discriminator (D). We also introduce the Recursive Cycle-Consistency Loss (R-CCL), which extends the cycle consistency loss to effectively preserve key anatomical structures and their spatial relationships, improving translation quality. Extensive experiments show that Reconfig-MIT is a generic framework that enables easy integration with existing image translation methods, including those incorporating registration networks used for correcting misalignments, and provides robust and high-quality translation on paired and unpaired misaligned data in both data-rich and data-limited scenarios.


### Impressive results

---
![Mian Figure](./figure/DF.png "Main Figure")

### Usage

---
Should you have any questions about using this repo, feel free to contact Jiahao Xu @ jiahaox@unr.edu

#### Example

To run a Re-SNGAN or Re-ProGAN model, you may follow:
1. Clone this repo to your local environment.
```
git clone https://github.com/IntellicentAI-Lab/Reconfig-MIT.git
```
2. Prepare the required and datasets. You may download the used dataset from: 
```
https://github.com/Kid-Liet/Reg-GAN
```
3. Run! For example:
```
# If you want to run a CycleGAN baseline on 10% of training data with noise level 5, you can use:

CUDA_VISIBLE_DEVICES=0 python train.py --model Reconfig_MIT --bidirect --cuda --data_ratio 0.1 --noise_level 5 --n_epochs 400

# If you want to run a RegGAN baseline on 10% of training data with noise level 5, you can use:

CUDA_VISIBLE_DEVICES=0 python train.py --model Reconfig_MIT --bidirect --cuda --data_ratio 0.1 --noise_level 5 --n_epochs 400 \
--regist   # The only difference is here.

# If you want to run a Reconfig-MIT  on 10% of training data with noise level 5, you can use:

CUDA_VISIBLE_DEVICES=0 python train.py --model Reconfig_MIT --bidirect --cuda --data_ratio 0.1 --noise_level 5 --n_epochs 400 \
--regist \   # The only difference is here.
--r_ccl --EL_lamda 10 \    # For R-CCL
--sparse --init-density 0.1 --final-density 0.7 --warmup_epoch 50 --final-grow-epoch 200 \
--update-frequency 1000 --method GMP --rm-first --regrow

```



# For ProGAN
python main.py --data_ratio 0.1 \
--regan --warmup_epoch 10 --g 10 --sparse 0.3 
# Please define epoch and batch size in the main.py