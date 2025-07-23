# Probing Cross-Modal Representational Alignment in Pretrained Vision and Language Models

This repository contains the code for the Neuromatch Academy NeuroAI course project by the team TyrOsine.

Recent advances in deep learning have revealed that model representations can resemble those found in the brain, particularly in sensory regions such as visual and auditory cortex (see Doerig et al. (2022) for a review). However, it remains underexplored which aspects of model design—modality, task specification and multimodal integration—significantly influence representational alignment with brain activity.

In this project, we will apply representational similarity analysis (RSA) between fMRI neural recordings obtained in the Natural Scenes Dataset (Allen et al., 2022) and model activations from pretrained vision, language and multimodal models. Our goal is to test how sensory input and task specification contribute to cognitively aligned representations, by comparing a range of architectures on image-caption stimuli. We are leveraging the Net2Brain toolbox (Bersch et al., 2025) for rapid experimentation and analysis towards this goal.

**Research questions:**

- How well do unimodal vision and language models align with brain representations during perception of images and language, respectively?

- Do multimodal vision-language models better explain brain responses than unimodal models during image presentation?

- How does task specification (e.g. autoencoding, semantic segmentation, next-token prediction) influence brain-model alignment across cortical regions?

**Setup**

To get started, clone the repository and run the setup script:

```bash
git clone https://github.com/emrecncelik/tyrosine.git
cd tyrosine
./setup.sh
```

This will create a `conda` environment named `net2brain`, and install the required packages.

