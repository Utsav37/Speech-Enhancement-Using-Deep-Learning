Speech Enhancement using different on training targets: IBM (Ideal Binary Mask), IRM (Ideal Ratio Mask), FFT-Mask and Clean Speech.

Input Data is Standardized and Normalized. For each of these, we have 4 Target Labels. 
  1. IBM (Ideal Binary Mask)
  2. IRM (Ideal Ratio Mask)
  3. FFT-Mask
  4. Clean Speech Signal (Correct speech label in Time Domain. It would have magnitudes of speech in time domain.)
  
Thus, in all we have 8 Deep Neural Networks in total. 4 [ Standardized Data (4 Target Labels) ] + 4 [ Normalized Data (4 Target Labels) ] 

**Input Data** : Noisy Speech
    Input Data Generation Jupyter Notebook
    How Noisy Data was prepared?
    For 1 clean speech: There were 3 noisy speeches and for each noisy speech, we combined that noisy speech with clean speech at 3           different desired db level (-3db,0db,3db)

Output Data: Predicted Mask for IBM,IRM,FFT-Mask and clean speech. For masked output predicted, we need to generate predicted clean speech with 
