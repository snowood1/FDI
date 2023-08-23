# Fake Document Infilling (FDI)

This repository contains the essential code for the paper [Controllable Fake Document Infilling for Cyber Deception (Findings of EMNLP 2022)](https://aclanthology.org/2022.findings-emnlp.486/). 

FDI is a controllable text-infilling model to generate realisitc fake copies of critical documents with moderate modification to protect the essential information and deceive adversaries.

## Folder

- FDI:     Proposed FDI inference pipeline
- ILM:   Text infilling model implementation modified from [1]
- WE_FORGE:  Reproduction of baseline [2]
- data: Our experimented datasets


## Quick Start

- Create training datasets with random masking.

      cd ILM
      sh create_datasets.sh


- Train a general text-infilling model.
    - See sample code in ILM/training_script.txt
    
- Inference via controllable masking.
    - See sample code in FDI/inference_demo.ipynb

## Evaluation details
- Check our [designed quiz, evaluated samples, and results.](https://docs.google.com/spreadsheets/d/11sayspimf_iDeXPZtI-lXI8iZxA74I8T/edit?usp=sharing&ouid=116390266661256212551&rtpof=true&sd=true)



## Reference

[1] Enabling language models to fill in the blanks. https://github.com/chrisdonahue/ilm

[2] Abdibayev, Almas, et al. "Using Word Embeddings to Deter Intellectual Property Theft through Automated Generation of Fake Documents." ACM Transactions on Management Information Systems (TMIS) 12.2 (2021): 1-22.




## Citation

If you find this repo useful in your research, please consider citing:

      @inproceedings{hu2022controllable,
        title={Controllable Fake Document Infilling for Cyber Deception},
        author={Hu, Yibo and Lin, Yu and Parolin, Erick Skorupa and Khan, Latifur and Hamlen, Kevin},
        booktitle={Findings of the Association for Computational Linguistics: EMNLP 2022},
        pages={6505--6519},
        year={2022}
      }

