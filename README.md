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

## Contact
Email : anandjohnbabyv4@gmail.com
