# Robust Continual Unlearning against Knowledge Erosion and Forgetting Reversal

This is the repository for the paper titled **Robust Continual Unlearning against Knowledge Erosion and Forgetting Reversal**.

## Abstract

> As a means to balance the growth of the AI industry with the need for privacy protection, machine unlearning plays a crucial role in realizing the "right to be forgotten" in artificial intelligence. This technique enables AI systems to remove the influence of specific data while preserving the rest of the learned knowledge. Although it has been actively studied, most existing unlearning methods assume that unlearning is performed only once. In this work, we evaluate existing unlearning algorithms in a more realistic scenario where unlearning is conducted repeatedly, and in this setting, we identify two critical phenomena:
(1) Knowledge Erosion, where the accuracy on retain data progressively degrades over unlearning phases, and
(2) Forgetting Reversal, where previously forgotten samples become recognizable again in later phases.
To address these challenges, we propose SAFER (StAbility-preserving Forgetting with Effective Regularization), a continual unlearning framework that maintains representation stability for retain data while enforcing negative logit margins for forget data. Extensive experiments show that SAFER mitigates not only knowledge erosion but also forgetting reversal, achieving stable performance across multiple unlearning phases.


## Results

### 1. Comparison of methods for continual unlearning 

- Unlearning Efficacy (ToW)
<table style="width: 100%; border-collapse: collapse;">
  <tr align="center">
    <td><img src="https://github.com/user-attachments/assets/d389c70c-4097-4c0d-b3dc-370907ef9e18" width="100%"><br><b> 1. CIFAR100 </b></td>
    <td><img src="https://github.com/user-attachments/assets/e565ca06-34a3-4cfe-acdd-eb982ed326d8" width="100%"><br><b> 2. VGGFace2 </b></td>
    <td><img src="https://github.com/user-attachments/assets/ab5a3755-9252-4c10-96b1-077ba4ec8014" width="100%"><br><b> 3. MUFAC </b></td>
  </tr>
</table>

- Knowledge Erosion
<img width="4170" height="1624" alt="KE_grouped_bar_f16_g3" src="https://github.com/user-attachments/assets/6ccc8781-2c65-4590-914e-9ace66b57e50" />


- Forgetting Reversal
<img width="4170" height="1624" alt="FR_grouped_bar_f16_g3" src="https://github.com/user-attachments/assets/9ebf0737-6f5e-4a51-9ffa-7ace863318d9" />


### 2. MIA results across phases
1. CIFAR100
<img width="2400" height="1200" alt="cifar2_mia_v4" src="https://github.com/user-attachments/assets/027580f7-1330-4d04-95f0-f762ae4718a7" />

2. VGGFace2
<img width="2400" height="1200" alt="vgg2_mia_v4" src="https://github.com/user-attachments/assets/96e51394-5222-4781-9011-8581e738480b" />

3. MUFAC
<img width="2400" height="1200" alt="mufac_mia_v4" src="https://github.com/user-attachments/assets/6d9354ec-e7c5-4d4d-9b87-f8183cdfa729" />

