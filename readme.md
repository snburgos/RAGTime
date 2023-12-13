# RAGTime

A benchmarking for RAG regarding time.

In `config` there are the instructions with and without RAG on english and chinese.

In `results` are the results from the databases. en stands for english and zh for chinese. Base means without RAG, RAG5 means with 5 documents (passage number) and RAG10 means with 10 documents. Inside each dataset there is a folder `time` that contains all the times for each model and each query.

Work based from [RGB](https://github.com/chen700564/RGB):

# RGB

- An implementation for [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431)

## Quick links

- [Environment](#Environment)
- [Retrieval-Augmented Generation Benchmark](#Retrieval-Augmented)
- [Evaluation](#Evaluation)

### Environment

```bash
conda create -n rgb python=3.10.0
conda activate rgb
bash env.sh
```

### Retrieval-Augmented Generation Benchmark

The data is putted in `data/`

```text
data/
├── en.json
├── en_int.json
├── en_fact.json
├── zh.json
├── zh_int.json
└── zh_fact.json
```

### Evaluation

For evaluating ChatGPT, you can run as:

```bash
python evalue.py \
--dataset en \
--modelname chatgpt \
--temp 0.2 \
--noise_rate 0.6 \
--api_key YourAPIKEY
```

For evaluating other models, you can run as:

```bash
python evalue.py \
--dataset en \
--modelname chatglm2-6b \
--temp 0.2 \
--noise_rate 0.6 \
--plm THUDM/chatglm-6b
```

You should change `modelname` and `plm` for different models, where `plm` is the path of model.

`temp` is the temperature of model.

`noise_rate` is rate of noisy documents in inputs.
