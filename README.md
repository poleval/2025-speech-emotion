# Polish Speech Emotion Recognition Challenge

## Introduction

**Speech emotion recognition** (SER) represents a critical area of research due to its extensive potential applications. Recent advancements in **automatic speech recognition** (ASR) and **large language models** (LLMs) have led to new possibilities for the development of SER systems. However, given that SER combines both audio processing and natural language understanding, the field still faces challenges. The subtle emotional cues are conveyed not only by what is said, but also by how it is said.

The difficulty of this task is caused by the subjective nature of emotions, both in their expression and perception. Each person may experience and interpret emotions differently, depending on factors such as language, cultural background, and situational context. Additionally, even subtle variations in speech can complicate generalization and reduce robustness, especially in the case of low-resource languages or challenging acoustic conditions.

In order to promote research in this area, we introduce the **Polish Speech Emotion Recognition Challenge**. The goal of this task is to evaluate how well current systems can identify emotional states from speech across diverse conditions, languages and speakers.

## Task Definition

The goal of this task is to develop an automatic system for classifying emotions based on audio recordings of human speech. Given an audio sample as input, the system should output a single label indicating the emotional state of the speaker.

This is a multi-class classification problem, with the following six target emotional states: **anger**, **fear**, **happiness**, **sadness**, **surprise**, and **neutral**.

Participants will receive a **train set** consisting of speech recordings in seven languages: **Bengali**, **English**, **French**, **German**, **Italian**, **Russian**, and **Spanish**. In addition, a **validation set** containing speech recordings in **Polish** will be provided for evaluation purposes.

The use of validation set for training - whether directly or indirectly (e.g. through data augmentation) - is **strictly prohibited**.

The **test set** will consist of previously unseen audio recordings of human speech in **Polish**. It is **forbidden** to manually label the test samples.

Participants may use **transfer learning** or **pretrained models**, provided that these models **have not been trained or fine-tuned** on Polish data or on **nEMO** dataset [[1]](#references).

Participants are required to work strictly within the provided dataset. The use of external resources or any additional data, including publicly available datasets, is **not allowed**.

## Dataset

### Dataset Format

Each set consists of:
- **Audio recordings**, which where resampled to 16kHz and saved as FLAC.
- **JSONL file** containing the following metadata for each sample:
    - `id`: A unique identifier for the audio sample corresponding to the filename.
    - `emotion`: The expected label for the sample.
    - `transcription`: The textual transcription of the speech.
    - `speaker_id`: An identifier for the speaker (`None` if not available).
    - `gender`: The gender of the speaker (`None` if not avaiable).
    - `age`: The age of the speaker (`None` if not available).
    - `dataset`: The name of the original dataset from which the sample was sourced.
    - `language`: The language spoken in the audio sample.

For **test set**, the expected labels are not provided.

### Train set

The train set consists of $29\,714$ audio recordings from 12 different datasets: **CaFE** [[2]](#references), **CREMA-D** [[3]](#references), **EMNS** [[4]](#references), **Emozionalmente** [[5]](#references), **eNTERFACE** [[6]](#references), **JL-Corpus** [[7]](#references), **MESD** [[8, 9]](#references), **Oreau** [[10]](#references), **PAVOQUE** [[11]](#references), **RAVDESS** [[12]](#references), **RESD** [[13]](#references), and **SUBESCO** [[14]](#references). 

The details of the language and distribution of samples per emotion in each subset, are shown in the table below.

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

### Validation set

The validation set consists of $4\,481$ audio recordings in Polish language from **nEMO** dataset [[1]](#references). The details of the distribution of samples per emotion, are shown in the table below.

|           | anger | fear  | happiness | neutral | sadness | surprise |
| --------- | ----- | ----- | --------- | ------- | ------- | -------- |
| # samples | $749$ | $736$ | $749$     | $809$   | $769$   | $669$    |

### Test set

> **TBA**

## Evaluation

The solutions will be evaluated based on two metrics:

$$ \text{Accuracy} = \frac{\sum_{i=1}^K \text{TP}_i}{N} $$

$$ \text{F1}_ \text{macro} = \frac{1}{K} \sum_ {i=1}^K \text{F1}_ i = \frac{1}{K} \sum_ {i=1}^K \left( \frac{2\cdot \text{Precision}_ i \cdot \text{Recall}_ i}{\text{Precision}_ i + \text{Recall}_ i} \right) = \frac{1}{K} \sum_ {i=1}^K \left( \frac{2\cdot \frac{\text{TP}_ i}{\text{TP}_ i + \text{FP}_ i} \cdot \frac{\text{TP}_ i}{\text{TP}_ i + \text{FN}_ i}}{\frac{\text{TP}_ i}{\text{TP}_ i + \text{FP}_ i} + \frac{\text{TP}_ i}{\text{TP}_ i + \text{FN}_ i}} \right) $$

where:
- $K$ - number of classes,
- $N$ - total number of samples,
- $\text{TP}_i$ - number of true positives for class $i$,
- $\text{FP}_i$ - number of false positives for class $i$,
- $\text{FN}_i$ - number of false negatives for class $i$.


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
