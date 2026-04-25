# CMAN — Manual INT8 symmetric quantization

---
**Given Data:**
<img width="653" height="257" alt="image" src="https://github.com/user-attachments/assets/5fd5d88f-0840-4202-8bbf-46356709646e" />

---
## a) Scale factor. Compute S using symmetric per-tensor quantization: S = max(|W|) / 127. Show the max value and the computed S.
-> <img width="785" height="262" alt="image" src="https://github.com/user-attachments/assets/37b82f10-16f7-4fbd-a42a-3444d0349aa1" />
  MAX Value - 2.32
  Scale Factor(S) = MAX ([W|])/127
               = 2.32/127
  **Scale Factor(S) = 0.01826772**
  
 ## b) Quantize. Quantize each element: W_q = round(W / S). Clamp to [−128, 127]. Write out the full 4×4 INT8 matrix.
 -> **Quantized**
 <img width="718" height="258" alt="image" src="https://github.com/user-attachments/assets/f0dc52f3-e921-4d63-a119-39aaaf1e6137" />

 ## c) Dequantized. Compute W_deq = W_q × S. Write out the 4×4 FP32 dequantized matrix.
 ->  **Dequantized**
 <img width="785" height="256" alt="image" src="https://github.com/user-attachments/assets/ab9fb4d3-caea-4924-ad2b-a975032a346e" />

 ## d) Error analysis. Compute the per-element absolute error |W − W_deq|. Identify the element with the largest error and compute the Mean Absolute Error (MAE) across all 16 elements.
 -> **Error Analysis**
 <img width="755" height="257" alt="image" src="https://github.com/user-attachments/assets/97d8d0c9-2763-448e-bc78-aa56aefbce61" />
 
 ## e) Bad scale experiment. Use S_bad = 0.01 (too small). Repeat quantization and dequantization. Compute the MAE. Explain in one sentence what goes wrong when S is too small.
 
