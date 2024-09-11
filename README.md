# RespLLM


This is the code used for the paper: RespLLM: Unifying Audio and Text with Multimodal LLMs for Generalized Respiratory Health Prediction.

This code utilize the [OPERA framework](https://github.com/evelyn0414/OPERA)[1] for the audio dataset downloading and processing, and you can put this directory under `src/benchmark`. 

example for training RespLLM:
```
python src/benchmark/RespLLM/RespLLM.py --llm_model OpenBioLLM --train_tasks S1,S2,S3,S4,S5,S6,S7 --test_tasks T1,T2,T3,T4,T5,T6 --train_epochs 3 --meta_val_interval 3  --train_pct 1 --batch_size 16 >> out_RespLLM_OpenBioLLM.txt
```



[1] Zhang, Yuwei, et al. "Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking." arXiv preprint arXiv:2406.16148 (2024).
