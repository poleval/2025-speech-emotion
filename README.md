# Polish Speech Emotion Recognition Challenge

## Introduction

**Speech emotion recognition** (SER) is a growing area of research with broad applications, driven
by advances in **automatic speech recognition** (ASR) and **large language models** (LLMs). SER is
uniquely challenging because it relies on both what is said and how it is said - capturing subtle
emotional cues through audio and language.

Emotions are subjective and influenced by cultural, linguistic, and contextual factors, making it
difficult to generalize across speakers and languages. These challenges are especially pronounced in
low-resource languages like Polish.

To address this, we present the **Polish Speech Emotion Recognition Challenge**. The goal is to
develop models that recognize emotions in Polish speech by effectively leveraging data from other
languages. This tests a system’s ability to generalize across linguistic and acoustic domains,
encouraging cross-lingual approaches to SER.

## Task Definition

The goal of this task is to build a system that classifies emotions from speech. Given an audio
recording, the system should predict one of six emotional states: **anger**, **fear**,
**happiness**, **sadness**, **surprise**, or **neutral**.

Participants will receive a **train set** consisting of speech recordings in seven languages:
Bengali, English, French, German, Italian, Russian, and Spanish. Additionally, a **validation set**
consisting of **Polish** speech will be provided for evaluation only. The use of validation set for
training or augmentation is strictly prohibited.

The final **test set** will consist of previously **unseen audio recordings of Polish speech**. It
is **forbidden** to manually label the test samples.

Participants may use **transfer learning** or **pretrained models**, provided that these models
**have not been trained or fine-tuned** on Polish data or on **nEMO** dataset [[1]](#references).

Participants are required to work strictly within the provided dataset. The use of external
resources or any additional data, including publicly available datasets, is **not allowed**.

## Dataset

The dataset used in this challenge is sourced from the **CAMEO** collection [[15]](#references),
which is a comprehensive collection of multilingual emotional speech corpora. The dataset is
available on Hugging Face: https://huggingface.co/datasets/amu-cai/CAMEO.

### Dataset Metadata and Structure

The **CAMEO** dataset provides rich metadata for each audio sample, as shown in the following example:

```python
{
  'file_id': 'e80234c75eb3f827a0d85bb7737a107a425be1dd5d3cf5c59320b9981109b698.flac', 
  'audio': {
    'path': None, 
    'array': array([-3.05175781e-05,  3.05175781e-05, -9.15527344e-05, ...,
       -1.49536133e-03, -1.49536133e-03, -8.85009766e-04]), 
    'sampling_rate': 16000
  }, 
  'emotion': 'neutral', 
  'transcription': 'Cinq pumas fiers et passionnés', 
  'speaker_id': 'cafe_12', 
  'gender': 'female', 
  'age': '37', 
  'dataset': 'CaFE', 
  'language': 'French', 
  'license': 'CC BY-NC-SA 4.0'
}
```

#### Data Fields

- `file_id` (`str`): A unique identifier for the audio sample.
- `audio` (`dict`): A dictionary containing:
    - `path` (`str` or `None`): Path to the audio file.
    - `array` (`np.ndarray`): Raw waveform of the audio.
    - `sampling_rate` (`int`): Sampling rate (16 kHz).
- `emotion` (`str`): The expressed emotional state.
- `transcription` (`str`): Orthographic transcription of the utterance.
- `speaker_id` (`str`): Unique identifier of the speaker.
- `gender` (`str`): Gender of the speaker.
- `age` (`str`): Age of the speaker.
- `dataset` (`str`): Name of the original dataset.
- `language` (`str`): Language spoken in the sample.
- `license` (`str`): License under which the original dataset is distributed.

### Download

All audio recordings and their corresponding metadata for this challenge are accessed directly
through the Hugging Face `datasets` library. Below is an example of how to load the dataset:

```python
from datasets import load_dataset

dataset = load_dataset("amu-cai/CAMEO", split="cafe")  # replace "cafe" with the desired split
```

### Train Set

For this challenge, the following splits of the **CAMEO** dataset are used in the **train set**:
`cafe`, `crema_d`, `emns`, `emozionalmente`, `enterface`, `jl_corpus`, `mesd`, `oreau`, `pavoque`,
`ravdess`, `resd`, and `subesco`.

The audio recordings and metadata are not provided directly in this repository. Instead, they are
accessed via the Hugging Face dataset. However, in the `in.tsv` file, each line specifies the split
name (corresponding to **CAMEO** splits) and the `file_id`, ensuring precise mapping between the
provided lists and the dataset hosted on Hugging Face. Example of the `in.tsv` file is shown below.

```
cafe	e9d4b7b83bd1f6825dabca3fc51acd62099b3ab70bd86f702495917b9a6541a9.flac
emozionalmente	2e4c53a24becdbf4f1b266439287f2e0d25d0bf29f0248e98480d19da62a97b1.flac
resd	ebcea26cf1ffffdb66eed7d7468b5ea9183ee41ac41941e59a1c51b15e4c41b6.flac
```

**Important Note:** This challenge does not use all samples from the original **CAMEO** dataset.
Only the samples representing relevant emotional states are included. These selected samples are
listed in the `in.tsv` file and used for the `train` set.

The train set consists of $29\,714$ audio recordings from 12 different datasets:
**CaFE** [[2]](#references), **CREMA-D** [[3]](#references), **EMNS** [[4]](#references),
**Emozionalmente** [[5]](#references), **eNTERFACE** [[6]](#references),
**JL-Corpus** [[7]](#references), **MESD** [[8, 9]](#references), **Oreau** [[10]](#references),
**PAVOQUE** [[11]](#references), **RAVDESS** [[12]](#references), **RESD** [[13]](#references),
and **SUBESCO** [[14]](#references).

The details of the language and distribution of samples per emotion in each subset, are shown in the
table below.

| Dataset        | Language | # samples | anger   | fear     | happiness | neutral | sadness  | surprise |
| -------------- | -------- | --------- | ------- | -------- | --------- | ------- | -------- | -------- |
| CaFE           | French   | $792$    | $144$    | $144$    | $144$    | $72$     | $144$    | $144$    |
| CREMA-D        | English  | $6\,171$ | $1\,271$ | $1\,271$ | $1\,271$ | $1\,087$ | $1\,271$ | -        |
| EMNS           | English  | $743$    | $133$    | -        | $158$    | $149$    | $150$    | $153$    |
| Emozionalmente | Italian  | $5\,916$ | $986$    | $986$    | $986$    | $986$    | $986$    | $986$    |
| eNTERFACE      | English  | $1\,047$ | $210$    | $210$    | $207$    | -        | $210$    | $210$    |
| JL-Corpus      | English  | $960$    | $240$    | -        | $240$    | $240$    | $240$    | -        |
| MESD           | Spanish  | $718$    | $143$    | $144$    | $144$    | $143$    | $144$    | -        |
| Oreau          | French   | $431$    | $73$     | $71$     | $72$     | $71$     | $72$     | $72$     |
| PAVOQUE        | German   | $4\,867$ | $601$    | -        | $584$    | $3\,126$ | $556$    | -        |
| RAVDESS        | English  | $1\,056$ | $192$    | $192$    | $192$    | $96$     | $192$    | $192$    |
| RESD           | Russian  | $1\,013$ | $219$    | $223$    | $218$    | $191$    | $162$    | -        |
| SUBESCO        | Bengali  | $6\,000$ | $1\,000$ | $1\,000$ | $1\,000$ | $1\,000$ | $1\,000$ | $1\,000$ |
| **Total**      |         | $29\,714$ | $5\,212$ | $4\,241$ | $5\,216$ | $7\,161$ | $5\,127$ | $2\,757$ |

### Validation Set

The **validation set** consists solely of the `nemo` split of the **CAMEO** dataset. As with the
training data, the audio recordings and metadata are accessible via Hugging Face.

The validation set consists of $4\,481$ audio recordings in Polish language from **nEMO** dataset
[[1]](#references). The details of the distribution of samples per emotion, are shown in the table
below.

|           | anger | fear  | happiness | neutral | sadness | surprise |
| --------- | ----- | ----- | --------- | ------- | ------- | -------- |
| # samples | $749$ | $736$ | $749$     | $809$   | $769$   | $669$    |

### Test Set

The audio recordings for the **test set** (`test-A` and `test-B`) are provided **directly** in this
repository (as `test-A.tar.gz` and `test-B.tar.gz` archives) and are not part of the **CAMEO**
dataset on Hugging Face. The corresponding metadata for these samples is included in a JSONL file
(`metadata.jsonl`), mirroring the structure of the metadata for `train` and `dev` sets, except for
the `emotion` label, which participants are expected to predict.

Example of the `metadata.jsonl` file is shown below.

```json
{
  "file_id":"bb7ee27f3e269c14b1b33538667ea806f20d7ba182cf9efe5d86a7a99085614f.flac",
  "transcription":"Ochronię cię.",
  "speaker_id":"SB0",
  "gender":"male",
  "age":"24",
  "dataset":"test",
  "language":"Polish",
  "license":"CC BY-NC-SA 4.0"
}
```

## Submission Format

The `out.tsv` file must contain exactly one label per line, corresponding to the emotional state
predicted for each audio sample. Each line in `out.tsv` should match the audio sample listed in the
same line of the `in.tsv` file, and should contain **only** the label, with no additional
information. Example of the `out.tsv` file is shown below.

```
neutral
anger
happiness
```

## Evaluation

The primary evaluation metric for this challenge is **macro-averaged F1 score** (**F1-macro**).
Additionally, overall **accuracy** will be reported as a secondary metric.

The F1-macro score is defined as:

$$ \text{F1}_ \text{macro} = \frac{1}{K} \sum_ {i=1}^K \text{F1}_ i $$

where $K$ is the number of classes, and $\text{F1}_i$ is the F1-score for class $i$, calculated as:

$$ \text{F1}_ i = \frac{2\cdot \text{Precision}_ i \cdot \text{Recall}_ i}{\text{Precision}_ i + \text{Recall}_ i} $$

with:

$$ \text{Precision}_ i = \frac{\text{TP}_ i}{\text{TP}_ i + \text{FP}_ i} $$

$$ \text{Recall}_ i = \frac{\text{TP}_ i}{\text{TP}_ i + \text{FN}_ i} $$

### Example

Given:

```python
y_true=[
    'happiness', 'happiness', 'neutral', 'surprise', 'neutral', 
    'happiness', 'sadness', 'sadness', 'fear', 'sadness',
]

y_pred=[
    'surprise', 'sadness', 'happiness', 'surprise', 'anger', 
    'happiness', 'anger', 'happiness', 'sadness', 'happiness',
]
```

The confusion matrix and intermediate metrics for each class are detailed in the table below.

| Emotion   | TP | FP | FN | TN | Precision            | Recall               | F1 score |
| --------- | -- | -- | -- | -- | -------------------- | -------------------- | -------- |
| anger     | 0  | 2  | 0  | 8  | $\frac{0}{0+2}=0$    | $\frac{0}{0+0}=0$    | $\frac{2\cdot 0\cdot 0}{0+0}=0$ |
| fear      | 0  | 0  | 1  | 9  | $\frac{0}{0+0}=0$    | $\frac{0}{0+1}=0$    | $\frac{2\cdot 0\cdot 0}{0+0}=0$ |
| happiness | 1  | 3  | 2  | 4  | $\frac{1}{1+3}=0.25$ | $\frac{1}{1+2}=0.33$ | $\frac{2\cdot 0.25\cdot 0.33}{0.25+0.33}=0.28$ |
| neutral   | 0  | 0  | 2  | 8  | $\frac{0}{0+0}=0$    | $\frac{0}{0+2}=0$    | $\frac{2\cdot 0\cdot 0}{0+0}=0$ |
| sadness   | 0  | 2  | 3  | 5  | $\frac{0}{0+2}=0$    | $\frac{0}{0+3}=0$    | $\frac{2\cdot 0\cdot 0}{0+0}=0$ |
| surprise  | 1  | 1  | 0  | 8  | $\frac{1}{1+1}=0.5$  | $\frac{1}{1+0}=1$    | $\frac{2\cdot 0.5\cdot 1}{0.5+1}=0.67$ |

The final F1-macro and accuracy:

$$ \text{F1}_ \text{macro} = \frac{1}{K} \sum_ {i=1}^K \text{F1}_ i = \frac{1}{6} \left( 0 + 0 + 0.28 + 0 + 0 + 0.67 \right) = \frac{1}{6} \cdot 0.95 = 0.1583 $$

$$ \text{Accuracy} = \frac{\sum_ {i=1}^K \text{TP}_ i}{N} = \frac{0+0+1+0+0+1}{10} = \frac{2}{10} = 0.2 $$

### Post-processing

Due to the generative nature of the LLMs, the models tend to generate descriptive responses instead
of a single-word output corresponding to the predicted emotional state. To ensure that the systems
are not penalized for minor errors, such as using the incorrect part of speech or responding with a complete sentence, we provide a script utilizing the post-processing strategy introduced in
[[15]](#references).

To use the script, run the following command:

```bash
python process_outputs.py <path_to_input_file> <path_to_output_file>
```

The `process_outputs.py` script takes two positional arguments - path to the input TSV file with the
outputs from a model, and path to the output TSV file, where the model's responses will be converted
to the labels corresponding to the emotional states, according to the following strategy: 

- If a generated response is not an exact match, it is normalized and split into a list of
individual words.

- Then, for each target label, the Levenshtein similarity score between the label and each word in
the generated response is computed.

- Similarity scores below a predefined threshold of 0.57 are filtered out for each label.

- The remaining scores are summed to yield an aggregated score for the given label.

- The label with the highest aggregated similarity score is selected as the best match from all
valid labels.

Additional details on the post-processing strategy, as well as an example, are available in
[[15]](#references).

## Baseline

### Qwen2-Audio-7B-Instruct

|          | `dev`  | `test-A` | `test-B` |
| -------- | ------ | -------- | -------- |
| F1-macro | 0.1829 | 0.1372   |          |
| Accuracy | 0.2160 | 0.1883   |          |

## References

1. I. Christop. 2024. [*nEMO: Dataset of Emotional Speech in Polish*](https://aclanthology.org/2024.lrec-main.1059/).

2. P. Gournay, O. Lahaie, and R. Lefebvre. 2018. [*A canadian french emotional speech dataset*](https://www.researchgate.net/publication/326022359_A_canadian_french_emotional_speech_dataset).

3. H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma. 2014. [*CREMA-D: Crowd-sourced emotional multimodal actors dataset*](https://www.researchgate.net/publication/272081518_CREMA-D_Crowd-sourced_emotional_multimodal_actors_dataset).

4. K. A. Noriy, X. Yang, and J. J. Zhang. 2023. [*EMNS /Imz/ Corpus: An emotive single-speaker dataset for narrative storytelling in games, television and graphic novels*](https://arxiv.org/abs/2305.13137).

5. F. Catania, J. W. Wilke, and F. Garzotto. 2020. [*Emozionalmente: A Crowdsourced Corpusof Simulated Emotional Speech in Italian*](https://www.researchgate.net/publication/388907774_Emozionalmente_A_Crowdsourced_Corpus_of_Simulated_Emotional_Speech_in_Italian).

6. O. Martin, I. Kotsia, B. Macq, and I. Pitas. 2006. [*The eNTERFACE’05 Audio-Visual Emotion Database*](https://aiia.csd.auth.gr/wp-content/uploads/papers/PUBLISHED/CONFERENCE/pdf/Martin06a.pdf).

7. J. James, L. Tian, and C. I. Watson. 2018. [*An Open Source Emotional Speech Corpus for Human Robot Interaction Applications*](https://www.isca-archive.org/interspeech_2018/james18_interspeech.pdf).

8. M. M. Duville, L. M. Alonso-Valerdi, and D. I. Ibarra-Zarate. 2021. [*The Mexican Emotional Speech Database (MESD): elaboration and assessment based on machine learning*](https://www.researchgate.net/publication/356919348_The_Mexican_Emotional_Speech_Database_MESD_elaboration_and_assessment_based_on_machine_learning).

9. M. M. Duville, L. M. Alonso-Valerdi, and D. I. Ibarra-Zarate. 2021. [*Mexican Emotional Speech Database Based on Semantic, Frequency, Familiarity, Concreteness, and Cultural Shaping of Affective Prosody*](https://www.researchgate.net/publication/356814365_Mexican_Emotional_Speech_Database_Based_on_Semantic_Frequency_Familiarity_Concreteness_and_Cultural_Shaping_of_Affective_Prosody).

10. L. Kerkeni, C. Cleder, Y. Serrestou, and K. Raoof. 2020.[*French Emotional Speech Database - Oréau*](https://doi.org/10.5281/zenodo.4405783).

11. I. Steiner, M. Schröder, and A. Klepp. 2013. [*The PAVOQUE corpus as a resource for analysis and synthesis of expressive speech*](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/7175).

12. S. R. Livingstone and F. A. Russo. 2018. [*The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391).

13. А. Аментес, И. Лубенец, and Н. Давидчук. 2022. [*Открытая библиотека искусственного интеллекта для анализа и выявления эмоциональных оттенков речи человека*](https://paperswithcode.com/dataset/resd).

14. S. Sultana, M. S. Rahman, M. R. Selim, and M. Z. Iqbal. 2021. [*SUST Bangla Emotional Speech Corpus (SUBESCO): An audio-only emotional speech corpus for Bangla*](https://journals.plos.org/plosone/article/file%3Ftype%3Dprintable%26id%3D10.1371/journal.pone.0250173).

15. I. Christop and M. Czajka. 2025. [*CAMEO: Collection of Multilingual Emotional Speech Corpora*](https://arxiv.org/abs/2505.11051).
