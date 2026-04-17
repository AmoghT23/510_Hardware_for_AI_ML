##CF03 CMAN — DRAM traffic analysis: naive vs. tiled matrix multiply <br>

---

Given Data - Two Square FP32 Matrices of size N x N (N = 32) stored and accessed in Row Major.  <br>

1. Naive Triple Loop (i j k) \
   There are N^2 output elements in C \
   So we have N^2 = 32^2 \
                  = 1024  <br>
   
   Total access in B matrix it would be N x N^2 \
   Total access in B matrix = N x N^2 \
                            = 32 x 1024 \
   Total access in B matrix = 32,768 access   <br>

   Total access for A and B matrix = 2 x N^3 \
                                   = 32,768 + 32,768 \
   Total access for A and B matrix = 65,536 access  <br>

   Traffic no reuse (every access is 4-byte fetch) \
   traffic = 65,536 access x 4byte \
   traffic(without C) = 262,144byte (256KiB) <br>
   
   ---
  2. Tiled Loop Analysis (T=8)\
     N=32 and T=8  so we divide each matrix into (N/T)^2 = 4 x 4  \
                                                    size = 16 tiles   \
                                                     T^2 = 64 elements (256 bytes)    \ <br>

     To compute 1 T x T tile of C we need to iterate through N/T pairs of A and B \
     For 1 C we need to load N/T tiles of A and N/T tiles of B \
     Total A tile loads = (N/T)^2 x (N/T) \
                        = (N/T)^3 \ 
                        = 4^3 \
                        = 64 tiles \ 
     As size of A and B is same, total tile loads of A and B is 64 \ <br>

      Total Tile Loads = T^2 = 64 elements \
      Total Tile Loads =  64 + 64

      Traffic with reuse (every access is 4-byte fetch) \
      traffic(each tile) = 64 access x 4byte \
      traffic(each tile) = 256 bytes/tile \ 
      traffic (for 64 tiles) = 2 x (256 x 64) \
      traffic(without C) = 2 x (16384) \
      traffic(without C) = 32,768byte (32KiB) \
      
   ---
   3. Ratio of naive DRAM traffic to tiled DRAM traffic.

      %% Naive
         ----- =
         Tiled
