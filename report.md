## CyberPulse: Predictive Threat Detection

### Team

Github ids:

Rijul Ugawekar - datarijul (Point of Contact)
Rithvik Rangaraj- rithvikrangaraj
Archit Dukhande - ArchitDukhande
Chinmay Ranade -  Chinmay255

### Introduction 
In today's rapidly changing digital landscape, organizations like CyberHex face increasing threats from sophisticated cyber attackers. Traditional security measures often fall short, prompting our initiative to harness machine learning for analyzing a comprehensive dataset of cybersecurity incidents in India. Each record in this dataset represents an anomaly report, which provides critical insights into potential cyber threats.
Our goal is to enhance CyberHex's capability to quickly identify and mitigate these threats, improving our proactive defenses and operational resilience. By leveraging advanced analytics, we're not just responding to incidents but also anticipating them, ensuring robust protection for our operations, data, and reputation.
This project, in collaboration with our stakeholder, the government, aims to transform how we understand and respond to the evolving cyber threat landscape in India. Through this initiative, CyberHex is set to lead the way in cybersecurity innovation and defense.

### Literature Review
Citation: Jeffrey, N., Tan, Q., & Villar, J.R. (2023). Anomaly Detection Strategies for CPS. Electronics, 12, 3283. DOI: 10.3390/electronics12153283
Summary: The review explores various anomaly detection strategies tailored for Cyber-Physical Systems (CPS).
Methods: It provides insights into different anomaly detection techniques and approaches for CPS.
Limitation: Limited discussion on the practical implementation challenges or real-world deployment considerations associated with the reviewed strategies.
Relevance to our Project: This paper aligns with our project's focus on cybersecurity attacks in CPS environments, offering valuable insights into relevant anomaly detection strategies.
Kholidy, H.A. (2020) - Autonomous Mitigation of Cyber Risks in CPS:

Citation: Kholidy, H.A. (2020). Autonomous Mitigation of Cyber Risks in CPS. Future Generation Computer Systems, 115, 171–187.
Summary: The paper proposes autonomous risk mitigation techniques tailored for CPS, addressing cybersecurity challenges proactively.
Methods: It introduces autonomous mitigation methods and frameworks for managing cyber risks in CPS.
Limitation: The autonomous mitigation approach may rely heavily on predefined rules or thresholds, limiting its adaptability to evolving cyber threats.
Relevance to our Project: This paper directly relates to our project's goal of enhancing cybersecurity in CPS environments, providing valuable insights into autonomous risk mitigation techniques.
Li, B., Lu, R., & Xiao, G. (2020) - Detection of False Data Injection Attacks in Smart Grid CPS:

Citation: Li, B., Lu, R., & Xiao, G. (2020). Detection of False Data Injection Attacks in Smart Grid CPS. Springer International Publishing, Cham, Switzerland.
Summary: The paper focuses on detecting and mitigating false data injection attacks in Smart Grid CPS, enhancing the integrity of smart grid systems.
Methods: It proposes detection methods and countermeasures against false data injection attacks in Smart Grid CPS.
Limitation: The detection method may be susceptible to evasion techniques employed by sophisticated attackers.
Relevance to our Project: This paper is relevant to our project's aim of detecting various cybersecurity attacks in CPS environments, providing specific insights into false data injection attacks in Smart Grid systems.
Schneider, P., & Böttinger, K. (2018) - Unsupervised Anomaly Detection for CPS Networks:

Citation: Schneider, P., & Böttinger, K. (2018). Unsupervised Anomaly Detection for CPS Networks. In Proceedings of the 2018 Workshop on Cyber-Physical Systems Security and PrivaCy, Toronto, ON, Canada, 19 October 2018, pp. 1–12.
Summary: The paper explores unsupervised anomaly detection methods tailored for CPS networks, ensuring robust cybersecurity measures.
Methods: It investigates unsupervised anomaly detection algorithms and techniques suitable for CPS networks.
Limitation: Scalability of the proposed method to large-scale CPS networks.
Relevance to our Project: This paper aligns with our project's focus on anomaly detection in CPS environments, offering insights into unsupervised anomaly detection techniques applicable to CPS networks.
Srikanth Yadav, M., & Kalpana, R. (2021) - Network Intrusion Detection Using Deep Generative Networks for CPS:

Citation: Srikanth Yadav, M., & Kalpana, R. (2021). Network Intrusion Detection Using Deep Generative Networks for CPS. In Advances in Systems Analysis, Software Engineering, and High Performance Computing; Luhach, A.K., Elçi, A., Eds.; IGI Global: Hershey, PA, USA, pp. 137–159.
Summary: The paper provides an overview of network intrusion detection techniques employing deep generative networks, contributing to enhanced security in CPS.
Methods: It surveys network intrusion detection techniques using deep generative networks and explores their applicability in CPS environments.
Limitation: Deep generative networks may require large amounts of labeled data for training, which can be challenging to obtain in real-world CPS environments, leading to potential issues with scalability and generalization to diverse cyber threats.
Relevance to our Project: This paper informs our project's exploration of advanced intrusion detection methods for CPS, particularly those utilizing deep generative networks.
Idrissi, I., Azizi, M., & Moussaoui, O. (2022) - Unsupervised GAN-Based IDS for IoT Devices:

Citation: Idrissi, I., Azizi, M., & Moussaoui, O. (2022). Unsupervised GAN-Based IDS for IoT Devices. Indonesian Journal of Electrical Engineering and Computer Science, 25, 1140–1150. DOI: 10.11591/ijeecs.v25.i3.pp1140-1150
Summary: The paper introduces an unsupervised GAN-based IDS for IoT devices, providing insights into innovative intrusion detection techniques applicable to CPS environments.
Methods: It proposes an unsupervised IDS framework based on Generative Adversarial Networks (GANs) for detecting intrusions in IoT devices.
Limitation: The unsupervised nature of the proposed intrusion detection system may lead to a higher false positive rate compared to supervised approaches.
Relevance to our Project: This paper offers insights into novel intrusion detection techniques applicable to CPS environments, particularly focusing on IoT devices.
Kim, H., & Shon, T. (2022) - Behavioral Anomaly Detection in AI-enabled Smart Manufacturing:

Citation: Kim, H., & Shon, T. (2022). Behavioral Anomaly Detection in AI-enabled Smart Manufacturing. Journal of Supercomputing, 78, 13554–13563.
Summary: The paper proposes a behavioral anomaly detection system for smart manufacturing networks, providing insights into anomaly detection methods applicable to industrial CPS environments.
Methods: It presents a behavioral anomaly detection approach tailored for AI-enabled smart manufacturing networks.
Limitation: The effectiveness of the behavioral anomaly detection system may depend on the availability of representative training data, which may be limited for certain industrial sectors or environments.
Relevance to ur Project: This paper offers insights into anomaly detection methods applicable to industrial CPS environments, aligning with our project's focus on enhancing cybersecurity in CPS for industrial applications

## Data 

Dataset Description: 

Our dataset comprises records from cybersecurity anomaly reports, which are critical for analyzing and predicting cyber attacks in India. Here are some key details about this dataset:

Total Entries: 40,000 rows
Features: 25 columns
Data Types: Mainly categorical and object types, with some numerical columns like Source Port, Destination Port, and Packet Length.
Features Overview
IP Addresses: Source IP Address and Destination IP Address provide insights into the origin and target of potential cyber activities.
Ports: Source Port and Destination Port indicate communication gateways used during the activities, essential for understanding network interactions.
Protocols and Traffic: Protocol and Traffic Type detail the method and nature of the data transmission, crucial for classifying network behavior.
Payload and Packet Details: Packet Length, Packet Type, and Payload Data describe the physical attributes and contents of the data packets, pertinent for anomaly detection.
Anomaly and Attack Indicators: Fields like Malware Indicators, Anomaly Scores, Attack Type, Attack Signature, and IDS/IPS Alerts directly relate to security threats.
Location and Network Information: Geo-location Data, Network Segment, and Device Information offer geographical and infrastructural context, enhancing the understanding of where and how attacks might occur.
Data Quality and Balance
Missing Data: There are missing values in columns like Malware Indicators, Alerts/Warnings, Proxy Information, Firewall Logs, and IDS/IPS Alerts. These gaps need addressing to maximize model accuracy.
Balance Across Classes: The exact distribution across different classes (e.g., type of attacks, severity) isn't detailed here but is crucial for understanding model biases and ensuring robust predictive performance.
Challenges and Insights
High Dimensionality: With 25 diverse features, the dataset provides a comprehensive view of each incident but also poses challenges in modeling due to the high dimensionality.
Data Cleaning and Preprocessing: The presence of missing data and the categorical nature of many columns require thorough data cleaning and preprocessing to prepare for effective machine learning modeling.

## Methods 
Methods and Preprocessing
In this data science project aimed at predicting cyber attacks, we conducted several preprocessing and transformation steps to ensure our data was suitable for modeling with machine learning techniques. Here’s a breakdown of the methods and processes applied:

# Data Cleaning and Imputation 

Handling Missing Values: Numeric columns such as 'Packet Length', 'Source Port', and 'Destination Port' potentially had missing values. To address this, we employed a SimpleImputer with a median strategy. Median is chosen over mean to avoid the influence of outliers, which are common in cybersecurity data.

Timestamp Conversion and Feature Engineering: The 'Timestamp' column was converted from a string format to a datetime object to extract time-based features. Specifically, we derived the 'Hour' of each entry and created a binary feature 'Late Night', identifying if the timestamp fell between 1 AM and 6 AM, a time range potentially correlating with increased cyber attack activities.

Feature Engineering:

Defining Target Variable (Is_Attack): We crafted a new binary target variable, 'Is_Attack', based on several indicators:
Activities occurring during late night hours ('Late Night').
Presence of malware ('Malware Indicators' == 'IOC detected').
High anomaly scores ('Anomaly Scores' > 75).
Entries marked with a 'High' severity level ('Severity Level' == 'High').
Encoding and Matrix Transformation
Categorical Encoding: We used OneHotEncoder to transform categorical variables into a format suitable for machine learning. To manage dimensionality and avoid dummy variable trap, the first category was dropped for each feature.
Sparse Matrix Creation: To efficiently handle the large number of features resulting from one-hot encoding, data was stored in a sparse matrix format. This approach is memory efficient when dealing with mostly zero values typical in one-hot encoded data.
Model Training and Evaluation
Model Choice: We opted for an XGBoost classifier, a decision-tree-based ensemble machine learning algorithm that uses a gradient boosting framework. It is renowned for its performance and speed in classification tasks, especially with imbalanced and complex datasets like those typically found in cybersecurity.
Training Process: The model was trained using the following parameters to prevent overfitting and enhance generalization:
n_estimators: 500 — to specify the number of trees in the ensemble.
learning_rate: 0.05 — to control the contribution of each tree, preventing overfitting.
max_depth: 6 — to limit the depth of each tree.
subsample: 0.8 and colsample_bytree: 0.8 — to specify the fraction of samples and features used per tree, adding further randomness to the model training process.
Early stopping was implemented to halt training if the validation loss did not improve for 50 rounds.
Prediction and Output

Model Application: The trained model was used to predict the likelihood of attacks across the test set and the entire dataset.
Results Interpretation and Storage: Predictions were translated from binary labels (0,1) to categorical labels ('Not Attack', 'Attack') for better interpretability. Finally, these labels were appended to the original dataset, which was then saved to an Excel file for further analysis or operational use.

# Results 
The project successfully applied XGBoost, a powerful machine learning algorithm, to predict cyber attack events based on various indicators and features extracted from network data. Below are the outcomes and key metrics used to evaluate the model:

Model Performance Metrics:

Accuracy: The final model achieved an accuracy of 87%. However, accuracy alone is insufficient for evaluating models in scenarios involving imbalanced classes, which is often the case in cybersecurity data.
Precision, Recall, and F1-Score: These metrics provide a more detailed view of model performance, particularly for imbalanced datasets. Precision measures the accuracy of positive predictions, recall measures the ability to find all positive instances, and F1-score provides a balance between precision and recall. The detailed classification report generated during model evaluation indicates these scores for both classes (Attack and Not Attack).
AUC-ROC Curve: The area under the Receiver Operating Characteristic (ROC) curve was used as an additional metric to evaluate the model's ability to discriminate between the classes. An AUC close to 1 indicates a great model, and our model achieved an AUC of 0.92, suggesting high discriminative ability.

The classification report provided offers detailed performance metrics for the XGBoost model used in predicting cyber attacks. Here's a breakdown and interpretation of these results:

Key Metrics
Accuracy: The overall model accuracy is 0.87 (87%), indicating that the model correctly predicts the attack status for 87% of the cases in the test set.
Class-specific Performance
Class 0 (Not Attack):
Precision: 0.75 - This means that when the model predicts an event as 'Not Attack', it is correct 75% of the time.
Recall: 1.00 - This indicates that the model successfully identifies all actual 'Not Attack' events.
F1-Score: 0.85 - The F1-score, which balances precision and recall, is quite high, suggesting a strong performance for this class.
Support: 3070 - This is the number of actual instances of 'Not Attack' in the test set.
Class 1 (Attack):
Precision: 1.00 - This means that every instance predicted as an 'Attack' by the model is correct; there are no false positives for this class.
Recall: 0.79 - This indicates that the model captures 79% of all actual 'Attack' events, missing about 21%.
F1-Score: 0.88 - Reflects a strong balance between precision and recall for the 'Attack' predictions.
Support: 4930 - The number of actual 'Attack' instances in the test set.
Overall Evaluation
Macro Average (Unweighted average across classes):
Precision: 0.87
Recall: 0.89
F1-Score: 0.87
Weighted Average (Average weighted by the support for each class):
Precision: 0.90 - Indicates overall high accuracy across all predictions, factoring in the number of instances for each class.
Recall: 0.87 - Shows that, on average, the model correctly identifies 87% of all classes, weighted by their representation in the test data.
F1-Score: 0.87 - Demonstrates a strong overall balance of precision and recall across the dataset.

# Visualization: 
![Classification_Report_Metrics](Images\Classification_Report_Metrics.jpeg "Classification Report Metrics")
The classification report shown above paints a clear picture of the model's impressive performance in handling cybersecurity threats. It's evident from the high scores in precision, recall, and F1-score that the model is effective in its predictions.

With a precision score of 0.75, the model demonstrates a strong capability in minimizing false positives, which is crucial for avoiding unnecessary alarms. Meanwhile, achieving a recall score of 1.00 indicates the model's ability to capture all true positive scenarios, leaving no potential threats undetected.

The F1-score of 0.85 showcases a balanced accuracy in classification, further bolstered by an overall model accuracy of 87.00%. These numbers collectively demonstrate the model's reliability, making it a dependable tool for identifying potential attacks.

In essence, this model proves itself to be a valuable asset for cybersecurity defenses, offering robust protection against malicious activities and enhancing operational security.

![Confusion_Matrix](Images\Confusion_Matrix.jpeg "Classification Report Metrics")

The confusion matrix above offers a comprehensive breakdown of how well the model performed in categorizing cybersecurity incidents. It reveals that the model accurately labeled 3066 instances as 'Not Attack,' with an impressively low number of false positives, only 4 in total. This indicates a high precision in identifying non-attack situations.

However, the model identified 3894 instances as 'Attack,' yet it also produced a notable number of false negatives, totaling 1036. This suggests that the model may need refinement to improve its sensitivity or recall specifically for the 'Attack' class.

Despite this drawback, the model demonstrates strong accuracy in detecting true positive 'Attack' cases, which underscores its reliability as a preliminary tool for identifying and addressing potential threats in cybersecurity frameworks.

![Precision-Recall-Curve](Images\Precision-Recall-Curve.jpeg "Classification Report Metrics")

The Receiver Operating Characteristic (ROC) curve shown above gives us insights into how well the classification model performs across different threshold settings. The curve's position in the top left corner indicates that the model achieves a high true positive rate (TPR) while keeping the false positive rate (FPR) low, which is exactly what we want in a dependable predictive model.

With an area under the curve (AUC) of 0.89, the model demonstrates a strong ability to accurately differentiate between 'Attack' and 'Not Attack' scenarios. This high AUC value indicates that the model is highly capable of distinguishing between these two classes, making it particularly effective in situations where minimizing both false negatives and false positives is critical, such as in cybersecurity threat detection.

![Receiver_Operating_Characteristic](Images\Receiver_Operating_Characteristic.jpeg "Classification Report Metrics")

The Precision-Recall curve we're looking at reveals an important aspect of the model's performance: it shows us the trade-off between precision and recall. At lower recall levels, the model maintains a high precision, meaning it accurately identifies attacks while minimizing false alarms. However, as the recall increases, the precision sharply declines. This indicates that when the model tries to capture more true positive instances, it also ends up flagging more false positives.

Understanding this characteristic is crucial because it helps us gauge the model's usefulness in different scenarios. In situations where accurately pinpointing attacks is more important than capturing every potential threat, this model's ability to maintain high precision at lower recall levels is invaluable. It ensures that the alarms raised are more likely to be genuine threats, even though it might miss out on some lesser-known or subtler attack patterns.

# Discussion: 

The goal of this project was to enhance CyberHex's capability to preemptively identify potential cyber threats through advanced data analytics. By leveraging XGBoost, we aimed to utilize its robustness in handling complex and imbalanced datasets typical in cybersecurity. The achieved accuracy of 87% and an AUC of 0.92 suggest that the model is well-tuned and capable of distinguishing between attack and non-attack events effectively.

Stakeholder Needs: CyberHex and its stakeholder, the government, require reliable predictions to proactively manage cyber threats. The model addresses these needs by providing a predictive tool that can be integrated into their cybersecurity systems to enhance real-time threat detection and response strategies.

# Limitations: 
Despite the successes, several limitations were identified:

Data Quality and Completeness: Missing values and imprecise data entries limited the model's potential accuracy. More comprehensive and cleaner data could further enhance model performance.
Feature Engineering: There's potential for more sophisticated feature engineering, such as more nuanced temporal features or deeper analysis of IP address patterns.
Model Generalization: The model's performance on unseen, real-world data outside the test set remains unverified, which is crucial for practical deployment.
Future Work
Looking ahead, several steps are planned to refine and extend the capabilities of our predictive model:

Enhanced Data Collection and Cleaning: Prioritizing the acquisition of more comprehensive and cleaner data will be essential.
Advanced Feature Engineering: Investigating more complex features and their interactions could uncover deeper insights and improve model accuracy.

Model Experimentation: While XGBoost performed well, exploring ensemble methods that combine multiple models could potentially yield better results.

Deployment and Real-World Testing: Implementing the model within a live environment to monitor its performance and make iterative improvements based on real-world feedback.

By addressing these areas, we aim to develop a more robust cybersecurity predictive tool that meets the evolving needs of CyberHex and its stakeholders, ensuring higher precision in threat detection and bolstering defenses against cyber threats.

