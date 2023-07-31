# Stable-CAT
### Stable Casual Attention Transformer(StableCAT) is a tiny, minimal modern LLM architecture for casual Language Modelling suitable for training medium size large language models and built for educational purposes
When parameters are scaled with good training data, Stable CAT can achieve comparable performance to existing models like gpt2 and even 3.5 with minor modifications. This was built for educational purpose and all scripts are kept minimal for easy reproducibility and training. Stable CAT leverages the decoder only architecture for its language modelling capabilities similar to exisiting models like Meta's Llama, Open Ai GPT-series etc.

# Training
The original model is trained with wikitext_v2 data. The reason for this is to allow sample training on consumer hardware and free resources. For better perfromance it's recommended you train Stable CAT on a much useful and larger dataset with scaled parameters. Currently there are no saved checkpoints due to limited hardware resources but we hope to release some in the future :)

## To train the model is as easy as running the training script and filling in the hyperprarameters
``` python train.py ```

# Maintainance and Contribution
Feel free to contribute and help maintain the repositories, submit pull requests and report any bugs found. I would really appreciate that.
