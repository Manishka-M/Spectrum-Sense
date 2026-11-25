SpectrumSense: AI-Powered RF Modulation Classification

SpectrumSense is a machine learning project that classifies four digital modulation schemes: BPSK, QPSK, 8PSK, and 16QAM, using synthetic complex-baseband signals under Additive White Gaussian Noise (AWGN). It uses statistical feature extraction, including amplitude characteristics and higher-order moments, followed by an RBF-kernel Support Vector Machine (SVM) classifier. The model was evaluated on an 80:20 stratified train-test split, achieving 87% test accuracy and 0.87 macro F1-score across 4,000 generated samples.

Tech Stack: Python, NumPy, Pandas, SciPy, Scikit-learn, Matplotlib, Streamlit
