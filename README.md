# SE3082 Lab 10: CUDA (Google Colab)

- `SE3082_Lab10_IT24101867.ipynb`: the Colab notebook for all four exercises. Upload it to https://colab.research.google.com (File, Upload notebook), switch the runtime to a T4 GPU, and use Runtime, Run all. Keep the outputs visible before you save or submit.
- `exercise-1`: timed programs (`ex1_timed.cu`, the timed Lab 9 programs, and `ex1c_whole_sequence.cu` for kernel-only vs whole-sequence time).
- `exercise-2`: `ex2_check.cu`, the Exercise 1 program with the `CHECK()` macro.
- `exercise-3`: `ex3_picture.cu`, PictureKernel on an 800 x 600 picture with a 2D grid.
- `exercise-4`: `ex4_volume.cu`, element-wise product of two 500 x 500 x 500 volumes with a 3D grid.

Compile any file with `nvcc -arch=sm_75 file.cu -o name && ./name` (use the `-arch` code that matches your GPU).

The sheet's self-check values are: Exercise 3 prints `Grid: 50 x 38 blocks of 16 x 16 threads` and 0 mismatches out of 480000; Exercise 4 prints `Grid: 63 x 63 x 63 blocks`, last result `C[499][499][499] = 47 x 68 = 3196`, and 0 mismatches out of 125000000.
