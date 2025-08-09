CUDA_VISIBLE_DEVICES=7 uv run dense_gemm_persistent.py                     \
      --mnkl 8192,8192,8192,2 --tile_shape_mnk 128,256,64                  \
      --cluster_shape_mn 2,1 --a_dtype Float8E4M3FN --b_dtype Float8E4M3FN \
      --c_dtype Float16 --acc_dtype Float32                           \
      --a_major k --b_major k --c_major n                                  \
      --warmup_iterations 100 --iterations 1000 --use_cold_l2                   \

CUDA_VISIBLE_DEVICES=7 uv run dense_gemm.py                     \
      --mnkl 8192,8192,8192,2 --tile_shape_mnk 128,256,64                  \
      --cluster_shape_mn 2,1 --a_dtype Float8E4M3FN --b_dtype Float8E4M3FN \
      --c_dtype Float16 --acc_dtype Float32                           \
      --a_major k --b_major k --c_major n                                  \
      --warmup_iterations 100 --iterations 1000 --use_cold_l2                   \