\# Third-Party Notices



\## PretrainedSED



This project incorporates and modifies code from \*\*PretrainedSED\*\*, developed by Florian Schmid and contributors.



\- \*\*Original repository:\*\* https://github.com/fschmid56/PretrainedSED

\- \*\*License:\*\* MIT License

\- \*\*License and copyright:\*\* See the original `LICENSE` file retained from PretrainedSED.



PretrainedSED provides pretrained audio models and supporting code for sound event detection, including model wrappers, training utilities, and data-processing components.



Parts of the original code have been adapted for research on respiratory sound classification using the ICBHI respiratory sound dataset.



\## Thesis-Specific Modifications



The thesis implementation includes changes to the original training pipeline, including:



\- Integration of an ICBHI dataset loader using dataset metadata and respiratory audio recordings.

\- Adaptation of the classification pipeline for binary Normal/Abnormal classification and, in relevant experiments, four-class classification.

\- Modification of model output handling and supervised loss computation for the target classification task.

\- Addition and modification of validation metrics, including accuracy, sensitivity, specificity, and harmonic score.

\- Addition of model checkpoint selection based on validation performance.

\- Addition of test-time evaluation and confusion matrix reporting.

\- In the later implementation, separation of training, validation, and test data, with patient-grouped splitting for the training and validation subsets.



These modifications were developed as part of a master's thesis on machine learning for abnormal respiratory sound detection. The original PretrainedSED code and the thesis-specific modifications should be understood as distinct contributions.





\## Other Third-Party Software and Model Assets



This project depends on third-party software libraries and may use pretrained model weights from PretrainedSED and other research projects. Each dependency and model asset remains subject to its own applicable license and terms.



The license of the PretrainedSED source code should not be assumed to automatically cover every pretrained checkpoint, externally sourced model, dataset, or other asset.



Before redistributing third-party weights or assets, consult the terms provided by their respective authors or distributors.



\## ICBHI Dataset



The ICBHI respiratory sound dataset is a third-party research dataset. Its recordings and associated metadata are not intended to be redistributed through this repository unless redistribution is permitted by the applicable dataset terms.



Users should obtain the dataset from its authorized source and follow its applicable conditions of use.



\## Research and Attribution



When using this project, please cite the relevant original PretrainedSED publication and any pretrained models or datasets used in an experiment, in addition to citing this thesis or its associated research publication where appropriate.



