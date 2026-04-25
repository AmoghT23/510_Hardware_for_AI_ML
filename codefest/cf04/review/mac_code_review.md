# MAC Unit LLM Code Review

| File | LLM |
|------|-----|
| mac_llm_A.sv | Gemini 3 (Fast) |
| mac_llm_B.sv | Claude 4.7 Opus |

---
## Simulation Using Questasim'

-> run.do file 
<img width="525" height="406" alt="image" src="https://github.com/user-attachments/assets/0c543b32-147f-4054-9fd2-b6d76f11da63" />

---

## mac_llm_A.sv — Simulation & Code Review
**DUT:** `mac_llm_A.sv` — INT8 MAC unit (AI accelerator building block)  
**Testbench:** `mac_tb.sv`

Transcript Output 
<img width="709" height="273" alt="image" src="https://github.com/user-attachments/assets/a34966b8-f060-47af-8a0f-a3800c2974bd" />

Waveform Output
<img width="1067" height="166" alt="image" src="https://github.com/user-attachments/assets/ac7811f8-d184-4ea2-9999-4ee22758a358" />

--- 

## mac_llm_B.sv — Simulation & Code Review
**DUT:** `mac_llm_B.sv` — INT8 MAC unit (AI accelerator building block)  
**Testbench:** `mac_tb.sv`

Transcript Output 
<img width="740" height="271" alt="image" src="https://github.com/user-attachments/assets/513f6cd6-421d-437b-ac05-431d581c50f1" />

Waveform Output

<img width="1065" height="162" alt="image" src="https://github.com/user-attachments/assets/4d6278fa-5f5e-40ed-b7c2-56ea4dd40765" />

---

## mac_correct.sv — Simulation & Code Review
**DUT:** `mac_correct.sv` — INT8 MAC unit (AI accelerator building block)  
**Testbench:** `mac_tb.sv`
**NOTE: 'mac_llm_B.sv' is the 'mac_correct.sv'**

Transcript Output 
<img width="733" height="273" alt="mac_correct_transcript" src="https://github.com/user-attachments/assets/ed410618-0ba5-438c-a9cd-99cf2661c608" />

Waveform Output 
<img width="1068" height="169" alt="mac_correct_wave" src="https://github.com/user-attachments/assets/2b5d7004-3575-4bca-a5a8-308c462d5323" />
