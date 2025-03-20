# DS-Net
Code for "DS-Net: A model driven network framework for lesion segmentation on fundus image".

[paper link](https://www.sciencedirect.com/science/article/pii/S0950705125002898)
![](https://github.com/tanfy929/DS-Net/blob/main/DS-Net.jpg)

We propose a framework named Decomposition-Segmentation Net (DS-Net) for the lesion segmentation task by unfolding EM algorithm operators into network modules step-by-step. D-Net focuses on separating lesions and background, while S-Net classifies lesion categories, making both sub-tasks easier than the original segmentation task. 

DS-Net can easily integrate with existing lesion segmentation networks in a plug-and-play manner, by setting them as the S-Net in the proposed framework. Here, examples are provided using UNet, L-seg, CE-Net, UNet++ and Swin-UNet as baseline segmentation networks, demonstrating how they can be integrated into the proposed DS-Net framework. The approach for replacing the S-Net is similar for any segmentation network.

**Usage**
> 
> - **`DataReader`**: Contains the ways to read IDRiD and DDR.
> - **`Models`**: Codes for networks used in paper (DS-UNet, DS-L-seg, DS-CE-Net, DS-UNet++, DS-Swin-UNet).
> - **`Tools`**: Codes for basic operations that may be used.
> - **`test_idrid.py`**: Codes for testing on IDRiD, similar to DDR. You can change the network model in *import* as needed (DS-UNet++ for example).
> - **`train_idrid.py`**: Codes for training on IDRiD, similar to DDR. You can change the network model in *import* as needed (DS-UNet++ for example).

**Citation:**
> Tan F, Wang Y, Xie Q, et al.  
DS-Net: A model driven network framework for lesion segmentation on fundus image[J].  
Knowledge-Based Systems, 2025: 113242.

**BibTeX:**
> @article{tan2025ds,
  title={DS-Net: A model driven network framework for lesion segmentation on fundus image},
  author={Tan, Feiyu and Wang, Yuhan and Xie, Qi and Fu, Jiahong and Wang, Renzhen and Meng, Deyu},
  journal={Knowledge-Based Systems},
  pages={113242},
  year={2025},
  publisher={Elsevier}
}