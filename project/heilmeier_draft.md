1. What are you trying to do? Articulate your objectives using absolutely no jargon. \
  -> I am building a custom hardware chip that analyzes microscope images of blood cells to detect anemia. Currently this requires either a trained specialist or a powerful computer. My chip performs the image pattern recognition step — the computationally heaviest part — on a small, low-power device. This enables a portable, battery-powered blood analyzer that a community health worker can use in a rural clinic with no internet, no lab, and no specialist — delivering a screening result in under one second.
   
2. How is it done today, and what are the limits of current practice? \
  -> Manual examination by a hematologist is slow, subjective, and unavailable in most resource-limited settings where anemia is most prevalent (affecting 2+ billion people globally). Automated approaches use CNNs on laptops/GPUs, achieving ~91% accuracy with hybrid models. But these consume 50–150W of power, making field deployment impractical. On a low-power MCU, the CNN inference takes 2–3 seconds per image because 2D convolution accounts for ~95% of the computation — too slow for screening dozens of patients.
   
3. What is new in your approach and why do you think it will be successful? \
->  Three things. First, I exploit the hybrid model's natural HW/SW split: lightweight handcrafted features (circularity, pallor, texture — 14 total) stay on the host MCU (~2ms), while the compute-heavy CNN convolutions go to a dedicated INT8 hardware accelerator. Second, my software baseline is pure NumPy — no PyTorch/TensorFlow — so every operation is profileable and the bottleneck is quantified with real data. Third, INT8 precision cuts multiplier area by ~4× versus FP32 with less than 1% accuracy loss. The accelerator uses output-stationary dataflow, connects via SPI, and targets a single convolutional block — a deliberately narrow scope that makes full synthesis and benchmarking achievable in 10 weeks.
   

   
