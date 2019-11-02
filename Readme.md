Speech Enhancement using different on training targets: IBM (Ideal Binary Mask), IRM (Ideal Ratio Mask), FFT-Mask and Clean Speech.

Input Data is Standardized and Normalized. For each of these, we have 4 Target Labels. 
  1. IBM (Ideal Binary Mask)
  2. IRM (Ideal Ratio Mask)
  3. FFT-Mask
  4. Clean Speech Signal (Correct speech label in Time Domain. It would have magnitudes of speech in time domain.)
  
Thus, in all we have 8 Deep Neural Networks in total. 4 [ Standardized Data (4 Target Labels) ] + 4 [ Normalized Data (4 Target Labels) ] 

**Input Data** : Noisy Speech
Input Data Generation Jupyter Notebook

**Output Data** : Predicted Mask for IBM,IRM,FFT-Mask and clean speech. For masked output predicted, we need to generate predicted clean predicted speech from our predicted mask by appropriate processing. 

Performance: Standardized Data with IRM,IBM and FFT-Mask performed good. 

File Description: 

1. Input Data Generation notebook: Creating Training Data (Created 9 training examples from original training example using three noise signals (3) and combining with every clean speech (1000) at desired signal to noise ratio of -3,0,3 db (3). (Thus, 3\*1000\*3=9000 training examples))
2. Data Preprocessing Notebook: Preparing Standardized and Normalized Data. 
3. Speech_Enhancement Notebook: All Deep Neural Network trained and tested here. 
