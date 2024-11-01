# 🎀 Breast Cancer Classification Project
This project presents a comparative analysis of different deep learning models for classifying breast cancer Histopathological images into two categories: benign and malignant. The models evaluated in this analysis include **DenseNet**, **VGG16**, **InceptionV3**, and **MobileNetV2**. The primary goal is to improve the accuracy of breast cancer diagnosis using a deep learning approach with Convolutional Neural Networks (CNNs).

# Dataset Description
The dataset used for this breast cancer classification project is the Breast Cancer Dataset available on Kaggle, contributed by *Anas Elmasry*. This dataset contains a collection of histopathological images that are crucial for the diagnosis of breast cancer. It includes various samples labeled as either benign or malignant, providing a solid foundation for training and evaluating machine learning models.

**Dataset Link** :  https://www.kaggle.com/datasets/anaselmasry/breast-cancer-dataset/d
### Dataset Sample Images
![ccca43c2-0104-4297-80bf-444b9aa33a7e](https://github.com/user-attachments/assets/4f48df59-07ec-41fd-b99f-933e4a7cab25)



## Model Architecture

The breast cancer classification project utilizes four different Convolutional Neural Network (CNN) architectures: **DenseNet**, **VGG16**, **InceptionV3**, and **MobileNetV2**. 

1. **Model Selection**: Each model is initialized using pre-trained weights from ImageNet, excluding the top layers to adapt them for our specific classification task of identifying benign and malignant images.
2. **Freezing Layers**: All layers of the base model are set to non-trainable, leveraging the pre-trained weights to prevent overfitting during training.
3. **Custom Layers**: The architecture is extended with custom layers, including a flattening layer, a dense layer with ReLU activation, a dropout layer for regularization, and a final dense layer with a softmax activation function to output probabilities for the two classes (benign and malignant).
4. **Compilation**: Each model is compiled using the **Adam** optimizer and **sparse categorical cross-entropy** loss function, with **accuracy** as the evaluation metric.

This architecture allows for effective feature extraction and classification, leveraging transfer learning from powerful pre-trained models.

## Evaluation Metrics and Visualizations
The **DenseNet** model demonstrated impressive performance on the test dataset, achieving an **accuracy of 84%**. This high accuracy reflects DenseNet's ability to effectively capture and learn complex patterns within the breast cancer images, making it a strong candidate for reliable classification between benign and malignant cases. The model's architecture, which emphasizes feature reuse and efficient gradient flow, contributes to its robustness in this classification task.


 ![84532604-5c6e-4888-8cf0-5d3ef80cbdf6](https://github.com/user-attachments/assets/acfd2d6f-19ba-435b-ac86-6d6c0b3939b7)
![c837bde3-4f62-4b26-920e-a4a38e4ae1cd](https://github.com/user-attachments/assets/c3897931-8244-4ffa-ba1d-792c19972320)

## Comparing Predictions from the Four Models

![1382bc19-9c98-4756-a40e-755fd3fb8797](https://github.com/user-attachments/assets/730a2558-e1c5-4274-a8f0-99a77f84ed67)

## Credits

### Dataset

This project uses the **Breast Cancer Dataset** available on Kaggle, created by **Anas Elmasry**. The dataset consists of histopathological images labeled as benign or malignant, which are used for training and evaluating machine learning models in this project.

- **Dataset Link**: [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/anaselmasry/breast-cancer-dataset/d)
- **Attribution**: Dataset © Original Authors (Anas Elmasry)
- **Usage Terms**: This dataset is copyrighted by the original authors and is subject to their usage terms as noted on the Kaggle page. Please refer to the dataset's Kaggle page for more details on its permitted uses.

### Pre-Trained Models

This project leverages several pre-trained Convolutional Neural Network (CNN) architectures for breast cancer classification. Each model is cited below with its original paper:

- **DenseNet**: Known for its dense connections between layers, DenseNet improves gradient flow and reduces the number of parameters.

  Citation: Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, 4700-4708. [Paper](https://arxiv.org/abs/1608.06993)

- **VGG16**: Developed by the Visual Geometry Group, VGG16 is known for its simplicity and uniform structure of convolutional layers, making it effective for many image classification tasks.

  Citation: Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556. [Paper](https://arxiv.org/abs/1409.1556)

- **MobileNetV2**: A lightweight model optimized for mobile and edge devices, with inverted residuals and linear bottlenecks.

  Citation: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, 4510-4520. [Paper](https://arxiv.org/abs/1801.04381)

- **ResNet50**: The ResNet architecture introduced residual connections to enable training of very deep networks, addressing vanishing gradient issues.

  Citation: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, 770-778. [Paper](https://arxiv.org/abs/1512.03385)

### Licensing

This repository is licensed under the MIT License, which applies only to the code within this project. The dataset used is not included under this license and remains subject to the terms and copyright of the original authors as outlined in the Dataset section above. For more details on the dataset's licensing and permitted uses, please refer to its [Kaggle page](https://www.kaggle.com/datasets/anaselmasry/breast-cancer-dataset/d).


## Contact

- **Email**: anandjohnbabyv4@gmail.com
