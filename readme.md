## Data-Efficient Alignment in Medical Imaging via Reconfigurable Generative Networks, WACV 2025


### Abstract

---
Recent advances in deep learning have witnessed many successful medical image translation models that learn correspondences between two visual domains. However, building robust mappings between domains is a significant challenge when handling misalignments caused by factors such as respiratory motion and anatomical changes. This issue is further exacerbated in scenarios with limited data availability, leading to a significant degradation in translation quality. In this paper, we introduce a novel data-efficient framework for aligning medical images via Reconfigurable Generative Network (Reconfig-MIT) for high-quality image translation. The key idea of Reconfig-MIT is to adaptively expand the generative network width within a Generative Adversarial Networks (GAN) architecture, initially expanding rapidly to capture low-level features and then slowing to refine high-level complexities. This dynamic network adaptation mechanism allows to adaptively learn at different rates, thus the model can better respond to deviations in the data caused by misalignments, while maintaining an effective equilibrium with the discriminator (D). We also introduce the Recursive Cycle-Consistency Loss (R-CCL), which extends the cycle consistency loss to effectively preserve key anatomical structures and their spatial relationships, improving translation quality. Extensive experiments show that Reconfig-MIT is a generic framework that enables easy integration with existing image translation methods, including those incorporating registration networks used for correcting misalignments, and provides robust and high-quality translation on paired and unpaired misaligned data in both data-rich and data-limited scenarios.


### Impressive results

---
![Mian Figure](./figure/DF.png "Main Figure")

### Usage

---

#### Hyperparameters introduction for Reconfig-MIT

##### General hyperparameters.

| Argument        | Type       | Description                                                               |
|-----------------|------------|---------------------------------------------------------------------------|
| `noise_level`         | int        | Noise level imposed to training data, ranging from 0-5                                           |
| `n_epochs`    | int        | Total training epochs          |
| `data_ratio`    | float      | To simulate a training data limited scenario                              |
| `regist`         | store_true | Enable registration network                                              |
| `r_ccl`      | store_true      | Enable R_CCL  |



##### Hyperparameters listed below are specifically for Reconfig-MIT.
| Argument        | Type       | Description                                                               |
|-----------------|------------|---------------------------------------------------------------------------|
| `reconfig_mit`             | store_true        | Enable Reconfi-MIT                                                      |
| `init-density`  | float        | Initial density  epochs                                                    |
| `final-density`  | float        | Final density to be growed epochs                                                    |
| `warmup_epoch`  | int        | Warmup training epochs                                                    |
| `final-grow-epoch`  | int        | The last epoch of growing                                                    |
| `update-frequency`  | int        | Grow frequency                                                    |
| `regrow`  | store_true        | Enable grow of the network                                                    |


#### Example

To run a Re-SNGAN or Re-ProGAN model, you may follow:
1. Clone this repo to your local environment.
```
git clone https://github.com/IntellicentAI-Lab/Reconfig-MIT.git
```
1. Prepare the dataset and place it in ./data/. You may download the used dataset from: 
```
https://github.com/Kid-Liet/Reg-GAN
```
1. Run! For example:
   
    3.1. If you want to run a CycleGAN baseline on 10% of training data with noise level 5, you can use:
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --model Reconfig_MIT --bidirect --cuda --data_ratio 0.1 --noise_level 5 --n_epochs 400
    ```

    3.2. If you want to run a RegGAN baseline on 10% of training data with noise level 5, you can use:
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --model Reconfig_MIT --bidirect --cuda --data_ratio 0.1 --noise_level 5 --n_epochs 400 \
    --regist   # The only difference is here.
    ```
    3.3. If you want to run a Reconfig-MIT  on 10% of training data with noise level 5, you can use:
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --model Reconfig_MIT --bidirect --cuda --data_ratio 0.1 --noise_level 5 --n_epochs 400 \
    --regist \   # Enable registration newtork
    --r_ccl --EL_lamda 10 \    # For R-CCL
    --sparse --init-density 0.1 --final-density 0.7 --warmup_epoch 50 --final-grow-epoch 200 \
    --update-frequency 1000 --method GMP --rm-first --regrow     # For Pruning
    ```


### Citation

If this work helps your research, please cite our paper.

### Acknowledgment

___
We would like to thank the work that help our paper:

1. Our code is builed up based on RegGAN: https://github.com/Kid-Liet/Reg-GAN.
2. GraNet: https://github.com/VITA-Group/GraNet.
