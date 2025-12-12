Tomato Leaf Disease Classification

A Comparative Study using MobileNetV2 and Support Vector Machines

This project focuses on building an image-based classification system to identify ten tomato leaf diseases using both deep learning and classical machine-learning approaches. The work evaluates how a fine-tuned MobileNetV2 model performs in comparison to an SVM classifier trained on MobileNet feature embeddings. The objective is to create a practical, accurate and lightweight solution that can support early disease detection in agricultural environments.

The dataset is derived from the PlantVillage Tomato Leaf collection and includes bacterial, fungal, viral and healthy leaf images. The images were cleaned, resized, normalised and split into training, validation and testing groups. MobileNetV2 was fine-tuned for end-to-end classification, while the SVM model utilised extracted bottleneck features. Both models were assessed using accuracy, precision, recall, F1-score, confusion matrices and ROC-AUC curves. Resource utilisation and inference behaviour were also examined to understand deployment suitability.

MobileNetV2 achieved the strongest performance, reaching approximately 98% accuracy and consistently high F1-scores across all classes. Its ability to learn detailed visual patterns resulted in fewer misclassifications and more stable predictions. The SVM model delivered respectable accuracy in the 91–93% range and showed excellent discrimination capability through high AUC values, although it struggled with classes that had similar symptoms or limited training samples.

The repository includes all scripts required for training, evaluation, feature extraction and visualisation. It also provides a Streamlit application that allows users to upload leaf images and obtain real-time predictions. Annotated outputs, performance plots and model artefacts are included to support transparency and usability.

This project demonstrates how deep learning and classical machine-learning methods can be combined to develop reliable crop-diagnostic tools. The solution is compact, efficient and suitable for integration into mobile or edge-based agricultural decision-support systems. It offers a strong foundation for future work involving field-quality datasets, ensemble models, explainable AI and deployment in real-world farming settings.
